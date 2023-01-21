import sys
import cx_Freeze

base = None

if sys.platform == "win32":
    base = "Win32GUI"
executables = [cx_Freeze.Executable("ExtractorAppPractice.py", base=base, icon="mybestplan_Jd2_icon.ico")]
cx_Freeze.setup(
    name = "ExtractAppTest",
    options = {"build_ex": {"packages": ["os", "shutil", "tkinter", "tkinter.ttk", "pytesseract", "numpy", "imutils", "cv2", "pdf2image", "namedtuple", "json", "tkPDFViewer"],
               "include_files": ["mybestplan_Jd2_icon.ico", "/Add_ons", "/Resources", "/Temporary"]}},
    version ="0.1",
    description = "test executable",
    executables = executables
    )

