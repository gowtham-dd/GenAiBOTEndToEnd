## File contains the code of the template created for this proj

import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO,format='[%(asctime)s]:%(message)s:')

project_name="GENAIBOTENDTOEND"
list_of_files=[
    ".github/workflows/.gitkeep",
    f"src/{project_name}/__init__.py",
    f"src/{project_name}/helper.py",
    f"src/{project_name}/prompt.py",
   
    ".env",
    "app.py",
    "requirements.txt",
    "setup.py",
    "research/trails.ipynb",
    
]



for filepath in list_of_files:
    filepath = Path(filepath)
    filedir,filename=os.path.split(filepath)

    if filedir !="":
        os.makedirs(filedir,exist_ok=True)
        logging.info(f"Creating directory;{filedir} for the file:{filename}")
    
    if(not os.path.exists(filepath))or (os.path.getsize(filepath)==0):
        with open(filepath,"w")as f:
            pass
            logging.info(f"Creating empty file:{filepath}")


    else:
        logging.info(f"{filename}File Already exists")