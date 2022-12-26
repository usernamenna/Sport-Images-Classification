import test
import  os

ROOT = os.path.dirname(os.getcwd())
TEST_PATH = ROOT+"/data"

test = test.Test(TEST_PATH)
test.predict()