import os
from IPython.lib import passwd

c.NotebookApp.ip = '0.0.0.0'
c.NotebookApp.port = 8888
c.NotebookApp.open_browser = False
# setting up the password
from IPython.lib import passwd
password = passwd("2525")
c.NotebookApp.password = password
c.MultiKernelManager.default_kernel_name = 'python36'