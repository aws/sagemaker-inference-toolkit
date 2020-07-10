import os
import signal
import subprocess
import sys

import pkg_resources
import psutil
from retrying import retry

import sagemaker_inference
from sagemaker_inference import environment, logging, utils
from sagemaker_inference.environment import code_dir

PYTHON_PATH_ENV = "PYTHONPATH"
logger = logging.get_logger()
REQUIREMENTS_PATH = os.path.join(code_dir, "requirements.txt")

def add_sigterm_handler(mms_process):
    def _terminate(signo, frame):  # pylint: disable=unused-argument
        try:
            os.kill(mms_process.pid, signal.SIGTERM)
        except OSError:
            pass

    signal.signal(signal.SIGTERM, _terminate)

    
def set_python_path():
    # MMS handles code execution by appending the export path, provided
    # to the model archiver, to the PYTHONPATH env var.
    # The code_dir has to be added to the PYTHONPATH otherwise the
    # user provided module can not be imported properly.
    code_dir_path = "{}:".format(environment.code_dir)

    if PYTHON_PATH_ENV in os.environ:
        os.environ[PYTHON_PATH_ENV] = code_dir_path + os.environ[PYTHON_PATH_ENV]
    else:
        os.environ[PYTHON_PATH_ENV] = code_dir_path

def install_requirements():
    logger.info("installing packages from requirements.txt...")
    pip_install_cmd = [sys.executable, "-m", "pip", "install", "-r", REQUIREMENTS_PATH]

    try:
        subprocess.check_call(pip_install_cmd)
    except subprocess.CalledProcessError:
        logger.error("failed to install required packages, exiting")
        raise ValueError("failed to install required packages")


# retry for 10 seconds
@retry(stop_max_delay=10 * 1000)
def retrieve_model_server_process(namespace):
    model_server_processes = list()

    for process in psutil.process_iter():
        if namespace in process.cmdline():
            model_server_processes.append(process)

    if not model_server_processes:
        raise Exception("model server was unsuccessfully started")

    if len(model_server_processes) > 1:
        raise Exception("multiple model servers are not supported")

    return model_server_processes[0]
