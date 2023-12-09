import importlib

def test_scipy_version():
    try:
        importlib.import_module('scipy')
    except ImportError:
        assert False, 'scipy module not found'

def test_numpy_version():
    try:
        importlib.import_module('numpy')
    except ImportError:
        assert False, 'numpy module not found'

def test_matplotlib_version():
    try:
        importlib.import_module('matplotlib')
    except ImportError:
        assert False, 'matplotlib module not found'

def test_pandas_version():
    try:
        importlib.import_module('pandas')
    except ImportError:
        assert False, 'pandas module not found'

def test_sklearn_version():
    try:
        importlib.import_module('sklearn')
    except ImportError:
        assert False, 'scikit-learn module not found'

def test_joblib_version():
    try:
        importlib.import_module('joblib')
    except ImportError:
        assert False, 'joblib module not found'
