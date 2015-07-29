import os
from distutils.core import setup
import py2exe

print "Removing Trash"
os.system("rmdir /s /q build")

# delete the old build drive
os.system("rmdir /s /q dist")


DLL_EXCLUDES = ['HID.DLL', 'MSVCP90.dll', 'libifcoremd.dll', 'w9xpopen.exe']

EXTRA_INCLUDES = ["email.iterators", "email.generator", "email.utils",
    "email.base64mime", "email", "email.mime", "email.mime.multipart",
    "email.mime.text", "email.mime.base", "lxml.etree", "lxml._elementpath",
    "gzip"]

EXCLUDES = ['IPython', 'Image', 'PIL', 'Tkconstants', 'Tkinter', '_hashlib',
    '_imaging', 'compiler', 'cookielib', 'doctest', 'email', 'matplotlib',
    'nose', 'optparse', 'pdb', 'pydoc', 'pywin', 'readline', 'sqlite3', 'tcl',
    'tornado', 'zmq']


setup(
    options={
        "py2exe":{
            # exclude these DLLs
            "dll_excludes": DLL_EXCLUDES,
            # compress the library archive
            "compressed": 0,
            #"skip_archive": 1,
            'includes': EXTRA_INCLUDES,
            "excludes": EXCLUDES,
            # lets try to make one EXE file
            'bundle_files': 1,
        }
    },
    console=[{'script': 'send_on_demand.py'}],
    data_files=[('.', ['send_on_demand.ini'])],
)
