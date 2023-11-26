import subprocess

def test_python_version():
    python_version_output = subprocess.check_output(['python', '--version'], text=True)
    assert 'Python' in python_version_output

def test_scipy_version():
    scipy_version_output = subprocess.check_output(['poetry', 'show', 'scipy'], text=True)
    assert 'scipy' in scipy_version_output

def test_numpy_version():
    numpy_version_output = subprocess.check_output(['poetry', 'show', 'numpy'], text=True)
    assert 'numpy' in numpy_version_output

def test_matplotlib_version():
    matplotlib_version_output = subprocess.check_output(['poetry', 'show', 'matplotlib'], text=True)
    assert 'matplotlib' in matplotlib_version_output

def test_pandas_version():
    pandas_version_output = subprocess.check_output(['poetry', 'show', 'pandas'], text=True)
    assert 'pandas' in pandas_version_output

def test_sklearn_version():
    sklearn_version_output = subprocess.check_output(['poetry', 'show', 'scikit-learn'], text=True)
    assert 'scikit-learn' in sklearn_version_output

def test_joblib_version():
    joblib_version_output = subprocess.check_output(['poetry', 'show', 'joblib'], text=True)
    assert 'joblib' in joblib_version_output
