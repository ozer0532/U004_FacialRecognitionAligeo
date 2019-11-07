import sys
from PyQt5.QtWidgets import QMainWindow, QApplication
from facegui import Ui_TubesAlgeo
from facerec import *

class AppWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_TubesAlgeo()
        self.ui.setupUi(self)
        self.show()  

    def generate(self, s):
        print("click", s)
        batch_extractor(dataset_path, self.ui)
        
    def test(self, s):
        print("click", s)

def batch_extractor(dataset_path, gui, pickled_db_path="features.pck"):
    dataset_files = [os.path.join(dataset_path, p) for p in sorted(os.listdir(dataset_path))]

    result = {}
    i = 0
    gui.progressBar.show()
    gui.progressBar.setProperty("value", i)
    length = len(dataset_files)
    for f in dataset_files:
        print ('Extracting features from image %s' % f)
        name = f.split('/')[-1].lower()
        result[name] = extract_features(f)
        i += 1
        gui.progressBar.setProperty("value", round(100*i/length))

    # saving all our feature vectors in pickled file
    with open(pickled_db_path, 'wb') as fp:
        pickle.dump(result, fp)

# DATA LOC
dataset_path = 'dataset/'
test_path = 'testset/'
dataset_files = [os.path.join(dataset_path, p) for p in sorted(os.listdir(dataset_path))]
test_files = [os.path.join(test_path, p) for p in sorted(os.listdir(test_path))]

# GUI
app = QApplication(sys.argv)
w = AppWindow()
w.show()
sys.exit(app.exec_())