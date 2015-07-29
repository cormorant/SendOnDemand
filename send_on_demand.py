#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
__version__="0.0.1.1"
COMPANY_NAME = 'GIN SB RAS'
APP_NAME = 'SendOnDemand'
"""
Программа отправки требуемых файлов с событиями.

Запуск программы производится каждый час встроенным средствами ОС (cron - linux,
планировщик заданий под Windows).

Требуется хранить список отправленных файлов в папках send(zip) (структура:
дата/время отправки, успешное/неудачное завершение отправки, имя файла).
Отправляться должны файлы только с временем события не больше 24 часов назад
(за последние сутки, вдруг не работал интернет, сайт seis-bykl и т.д.).

Есть папка с данными, в ней по запросу времени в виде YYYYMMDDTHHMMSS требуется
найти файлы в формате ХХ (Байкал-5), при необходимости объединить/вырезать нужный
участок с данными (по умолчанию 1минута 30 секунд назад и столько же вперед, итого 3 минуты).

Окончательное ТЗ:
    - скачивание файла запроса (текстовый файл на ФТП в папке с именем станции;
    формат файла -- строчки YYYYMMDDTHHMMSS)
    - поиск нужного участка в файлах непрерывки
    - вырезка и сохранение результирующего файла
    - отправка файла на фтп/по почте.

TODO:
    - длина файла м.б. не везде одинакова! Т.е. не 5 минут, а где-то меньше...
    - убрать AssertionError если больше 2-х файлов. Просто брать первые 2
"""
import os
import sys
import copy
import datetime; now = datetime.datetime.now
import ftplib
import tarfile
import smtplib
import email
from urllib2 import urlopen
import struct
from BaikalFile import BaikalFile
import ConfigParser
# instead of Obspy
from myobspy.obspycore.utcdatetime import UTCDateTime
from myobspy.obspycore.trace import Trace
from myobspy.obspycore.stream import Stream
import numpy as np


def module_path():
    if hasattr(sys, "frozen"):
        return os.path.dirname(sys.executable)
    return os.path.realpath(os.path.dirname(__file__))

# путь откуда запущен скрипт
CurrDir = module_path()
# путь к папке с выходными файлами
OutDir = os.path.join(CurrDir, "out")
# путь к папке с логами
LogDir = os.path.join(CurrDir, "log")
# путь к папке с текстовыми запросами
REQUESTS_PATH = os.path.join(CurrDir, "requests")

# create dirs if needed
for Dir in (OutDir, LogDir, REQUESTS_PATH):
    if not os.path.exists(Dir): os.makedirs(Dir)

sys.stdout = open(os.path.join(LogDir, "send_on_demand.log"), "a")
sys.stderr = open(os.path.join(LogDir, "send_on_demand.err"), "a")

# where to find config?
CONFIG_FILENAME = os.path.join(CurrDir, "send_on_demand.ini")
#===========================================================================
#=== read Settings from config file
config = ConfigParser.SafeConfigParser()
config.read(CONFIG_FILENAME)
# read main options
section = "main"
if not config.has_section(section):
    print("No section '{}' in config file {}! Exiting.".format(section,
        CONFIG_FILENAME))
    sys.exit(0)
# read all config in dictionary (default type is str)
Settings = dict( (k, v) for k, v in config.items(section) )
# make int
for key in ("after", "before"):
    Settings[key] = int(Settings[key])
# read email config section
section = "email"
if not config.has_section(section):
    print("No section '{}' in config file {}! Exiting.".format(section,
        CONFIG_FILENAME))
    sys.exit(0)

EmailSettings = dict( (k, v) for k, v in config.items(section) )
EmailSettings['debuglevel'] = int(EmailSettings['debuglevel'])
EmailSettings['email_to'] = EmailSettings['email_to'].split()
#===========================================================================

# ISO8601 DateTime format
DateTimeFormats = ("%Y-%m-%dT%H:%M:%S", #ISO8601
    "%Y%m%dT%H:%M:%S", "%Y%m%dT%H%M%S", "%Y%m%d%H%M%S")

#TODO: 'out' also must be variable
# формат выходного файла для записи
FILENAME_FORMAT = os.path.join(OutDir, "{0.year:04d}{0.month:02d}{0.day:02d}_"
    "{0.hour:02d}{0.minute:02d}{0.second:02d}")

# каналы
CHANNELS = ("N-S", "E-W", "Z")


def isconnected():
    """ Проверка подключения к интернету """
    reliableservers = ("http://84.237.36.66", "http://rambler.ru", "http://google.com")
    connected = False
    for reliableserver in reliableservers:
        try:
            urlopen(reliableserver)
        except:# IOError, URLError:
            return False
        else:
            return True


def tar_file(localFile):
    """ упаковка файла bzip-ом. Возвращает имя созданного архива """
    path, filename = os.path.split(localFile)
    tarfilename = os.path.join(path, filename + ".tar.bz2")
    #if os.path.exists(tarfname): return
    tar = tarfile.open(tarfilename, "w:bz2")
    tar.add(localFile, arcname=filename)
    tar.close()
    return tar.name


def send_email(**kwargs):
    """ sending email with pure python """
    # create message
    msg = email.MIMEMultipart.MIMEMultipart()
    msg['Subject'] = kwargs['subject']
    msg['From'] = kwargs['email_from']
    msg['To'] = ', '.join(kwargs['email_to'])
    # attach or not?
    filename = kwargs["filename"]
    if filename is not None and os.path.exists(filename) and os.path.isfile(filename):
        part = email.MIMEBase.MIMEBase('application', "octet-stream")
        part.set_payload(open(filename, "rb").read())
        email.Encoders.encode_base64(part)
        part.add_header('Content-Disposition',
            'attachment; filename="{}"'.format(os.path.split(filename)[-1]))
        msg.attach(part)
    # connect to smtp server
    server = smtplib.SMTP(kwargs['email_server'])
    server.set_debuglevel = kwargs['debuglevel']
    # secure mode
    server.starttls()
    server.login(kwargs['username'], kwargs['password'])
    # sending email
    server.sendmail(kwargs['email_from'], kwargs['email_to'], msg.as_string())
    server.quit()
    return True


def download_new_requests(ftphost, ftpuser, ftppass, ftpdir, **kwargs):
    """ поиск на ФТП файлов с запросами и сохранение новых в папке requests"""
    print("{}\tConnecting to host {}, directory {}".format(now(), ftphost, ftpdir))
    # подключиться к ФТП (lspserver)
    conn = ftplib.FTP(host=ftphost, user=ftpuser, passwd=ftppass)
    # получить файлы в папке СТАНЦИЯ
    conn.cwd(ftpdir)
    # получить список файлов с запросами в удаленной папке
    try:
        files = conn.nlst()
    except ftplib.error_perm, resp:
        if str(resp) == "550 No files found":
            print("{}\tNo files in directory {}".format(now(), ftpdif))
            return
    # скачать все файлы, которые еще не скачаны
    downloaded_files = []
    for rfile in files:
        # save file into RequestsDir
        filename = os.path.join(REQUESTS_PATH, rfile)
        if os.path.exists(filename):
            print("{}\tSkipping existing file {}".format(now(), rfile))
            #continue# or rewrite? file may have same name
            # overwrite file, because may be file with same name but other content!
        _f = open(filename, "w")
        try:
            conn.retrbinary('RETR ' + rfile, _f.write)
        except ftplib.error_perm:
            print("Error downloading remote file {}".format(rfile))
            continue
        finally:
            _f.close()
        #print("Append {}".format(filename))
        downloaded_files += [filename]
    else:
        print("{}\tSuccesfully downloaded {} file(s).".format(now(), len(downloaded_files)))
    return downloaded_files


def get_dt_from_requestfile(filename):
    """ поиск в текстовых файлах сточек с датой/временем """
    #=== в файле найти дату/время требуемого события
    # получить строки из файла
    with open(filename) as _f: lines = [s.strip() for s in _f.readlines()]
    # для каждой строки получить дату/время
    datetimes = []
    for line in lines:
        # try with every format of DateTime
        for dtFormat in DateTimeFormats:
            try:
                dt = datetime.datetime.strptime(line, dtFormat)
            except ValueError:
                continue
            else:
                datetimes += [dt]
                break
    return datetimes


def create_trace(header, channel, data):
    """ create stream from header and data on given channel """
    date = datetime.datetime(*[header[k] for k in ("year", "month", "day")])
    delta = datetime.timedelta(seconds=header["to"])
    dt = date + delta
    # make utc datetime
    utcdatetime = UTCDateTime(dt, precision=3)
    # подготовить заголовок
    stats = {
        'network': "NT",
        'location': "LOC",
        #"calib": 1.0,
        "station": header['station'].upper()[:3],
        'channel': channel,
        'sampling_rate': round( 1./header["dt"] ),
        "delta": header["dt"],
        "npts": data.size,
        'starttime': utcdatetime,
    }
    # создать трассу
    trace = Trace(header=stats, data=data)
    return trace


def search_baikal_files(start, end, path):
    """ индексировать файлы формата Байкал-5 в указанной папке. Оставлять только
    те файлы, которые попадают в промежуток start..end """
    result = []
    files = [os.path.join(path, fil) for fil in os.listdir(path)]
    files.sort()
    for filename in files:
        if not os.path.isfile(filename): continue
        # чтение файла и запись его времени
        bf = BaikalFile(filename)
        if not bf.valid: continue
        # считаем из файла в формате Байкал дату/время начала этого файла
        dt = bf.get_datetime()
        # сколько секунд длительность файла
        seconds = bf.samples_count * bf.MainHeader["dt"]
        dt_end = dt + datetime.timedelta(seconds=seconds)
        # если файл попадает
        if (dt <= start <= dt_end) or (dt <= end <= dt_end):
            # save into list filename + bf (BaikalFile object)
            result.append( [filename, bf] )
    return result


def read_merge_and_write(start, end, values):
    """ Функция считывания, обрезки и записи выходного файла.
    Параметры: требуемое начало и конец файла, список """
    # collect data from files
    traces = []
    for filename, bf in values:
        # stream for every trace
        for channel, data in zip(CHANNELS, bf.traces):
            trace = create_trace(bf.MainHeader, channel, data)
            traces += [trace]
    stream = Stream(traces) # create stream
    stream.merge(method=1)  # merge it!
    # trim stream
    stream.trim(starttime=UTCDateTime(start), endtime=UTCDateTime(end))
    stream.sort()           # sort stream
    # now if E-W is 1st, move it
    if stream[0].stats.channel == "E-W" and stream[1].stats.channel == "N-S":
        # поменять местами
        stream[0], stream[1] = stream[1], stream[0]
    #=== Write resulting file
    # имя для нового файла (время, станция)
    outfilename = FILENAME_FORMAT.format(start) + \
        ".{}".format(bf.MainHeader['station'][:3].lower())
    # если такой файл уже сушествует, значит этот запрос мы уже обработали...
    if os.path.exists(outfilename): return
    _f = open(outfilename, "wb")
    # пишем заголовки целиком
    nkan = len(stream)
    bytes_to_read = 120 + nkan * 72
    with open(values[0][0], "rb") as f1: header = f1.read(bytes_to_read)
    _f.write(header)
    # дальше допишем данные
    a = np.array([trace.data for trace in stream])
    _f.write( a.T.flatten().tostring() )
    # изменить соответствующую 1-ю секунду в заголовке (at 56)
    if start != stream[0].stats.starttime.datetime:
        print("Warning: start times mismatch: {} and {}!!!".format(start,
            stream[0].stats.starttime.datetime))
    _f.seek(56)
    first_sec = (start - datetime.datetime.combine(start.date(),
        datetime.time(0))).total_seconds()
    _f.write( struct.pack('d', first_sec) )
    _f.close()
    # return written file name(s)
    return outfilename


def do_the_job(dt, PATH):
    """ искать требуемый участок данных в непрерывке """
    date = dt.date()
    print("{}\tSearching for day {}".format(now(), date))
    #=== в папке PATH у нас лежат файлы в виде _MM_DD
    # найдем нужную папку с данными за этот день
    day_path = "_{:02d}_{:02d}".format(date.month, date.day)
    # общий путь к файлам за этот день
    path = os.path.join(PATH, day_path)
    if not os.path.exists(path):
        print("{}\tNo data for day {}".format(now(), date))
        return
    # для каждого времени из переданного списка найти попадающие туда файлы
    print("{}\tSearching for time {}, day is {}".format(now(), dt.time(), date))
    # требуемое начало и конец файла
    start = dt - datetime.timedelta(seconds=Settings["before"])
    end = dt + datetime.timedelta(seconds=Settings["after"])
    # search all files in path, but leave only files that match in (start..end)
    result_data = search_baikal_files(start, end, path)
    #=== считывание, обрезка и запись выходного файла
    try:
        result = read_merge_and_write(start, end, result_data)
    except BaseException as e:
        print("Error working with files: {}".format(e))
        result = None
    #=== записали файл. Теперь его упаковать и отправить
    if result is None: return
    # упаковка файла (tar.bz2)
    tarresult = tar_file(result)
    # отправка файла
    kwargs = copy.copy(EmailSettings)
    # attach file
    kwargs["filename"] = tarresult
    # run sending function
    try:
        succesful = send_email(**kwargs)
    except BaseException as e:
        print("Error sending email: {}".format(e))
        succesful = False
    print("{}\tSending packed file {}... {}".format(now(), result,
        "Succesful" if succesful else "Failed"))


def main():
    print("*"*77)
    #===========================================================================
    # проверка наличия папки с данными
    if not os.path.exists(Settings["path"]):
        print("Cannot find path '{}'! Exiting.".format(Settings["path"]))
        sys.exit(1)
    #===========================================================================
    # проверка соединения с интернетом
    if not isconnected():
        # установить соединение с интернетом (dial-up via modem)
        print("%s\tConnecting to internet." % now())
        #...
    else:
        print("%s\tAlready connected to internet." % now())
    #===========================================================================
    #=== по всем файлам запросов (TXT) составить список требуемых дней-времен
    # поиск на ФТП файлов с запросами
    files_requests = download_new_requests(**Settings)
    # получить список всех дней/времен по всем файлам
    datetimes = []
    for request_file in files_requests:
        # получить из текст. файла список дат/времен, для которых искать события
        next_datetimes = get_dt_from_requestfile(request_file)
        if datetimes is not None:
            datetimes += next_datetimes
    # Group datetimes by date (to avoid indexing same directory many times)
    datetimes.sort()
    #=== now index and search; искать данные
    # в каждой папки с суточными данными искать требуемый участок данных
    for dt in datetimes:
        result = do_the_job(dt, Settings["path"])
    #===========================================================================
    print("*"*77)
    return 0


if __name__ == '__main__':
    main()

