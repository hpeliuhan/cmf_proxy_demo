from cmflib import cmf
import subprocess
import os
import time
import asyncio
import tarfile
from pathlib import Path
from datetime import datetime
import threading
import shutil
from waggle.plugin import Plugin
from waggle.plugin import get_timestamp
import logging
import hashlib

logger = logging.getLogger(__name__)



class cmfsage:
    def __init__(self,pipeline_name,pipeline_file,model_path,result_path,git_remote_url,archiving=True,logging_interval=5):
        self.pipeline_name= pipeline_name
        self.pipeline_file= pipeline_file
        self.model_path= model_path
        self.result_path= result_path
        self.git_remote_url= git_remote_url
        self.archiving= archiving #if we are archiving the result, we put the arvhive file into the logging queue otherwise we put the result intot the logging queue
        self.logging_interval= logging_interval
        self.monitoring_interval=0.05
        self.result_archiving_interval= 2
        self.stop_event=None
        self.result_backup_dir="result_backup_dir"
        self.result_archive_dir = "result_archives"  
        self.cmf_archive_path = "cmf_archives"
        self.result_archiving_queue= asyncio.Queue()
        self.cmf_logging_queue = asyncio.Queue() 
        self.cmf_archive_queue = asyncio.Queue()
        self.cmf_upload_queue = asyncio.Queue() 

    def check_path(self):
        # If path exists, use os.path.isdir/os.path.isfile
        if os.path.exists(self.result_path):
            if os.path.isdir(self.result_path):
                return self.result_path, "dir"
            elif os.path.isfile(self.result_path):
                return os.path.dirname(self.result_path), "file"
        # If path does not exist, use heuristic: if it has a file extension, treat as file
        _, ext = os.path.splitext(self.result_path)
        if ext:
            return os.path.dirname(self.result_path), "file"
        else:
            return self.result_path, "dir"
    
    def relative_path(self,filepath):

        return os.path.relpath(filepath)
    
    async def setup(self):
        '''Check the path is diretory or file'''
        '''If it is a directory, we will use the directory as the result path'''
        '''If it is a file, we will use the parent directory as the result path'''
        
        #check if archiving
        if self.archiving:
            #self.result_archiving_path="result_archives"
            os.makedirs(self.result_archive_dir, exist_ok=True)
            
        #result path check
        #self.result_path, self.result_type = self.check_path()
        self.result_dir, self.result_type = self.check_path()

        await asyncio.gather(
            self.monitor_and_archive(),
            self.process_archives(),
            self.log_result_archives(),
            self.cmf_archive(),
            self.cmf_upload(),
            self.log_heartbeat()  
        )

    async def process_archives(self):
        while True:
            archive_paths = []
            while not self.result_archiving_queue.empty():
                logger.info("Processing result archiving queue...")
                archive_path = await self.result_archiving_queue.get()
                archive_paths.append(Path(archive_path))
                self.result_archiving_queue.task_done()
            if archive_paths:
                os.makedirs(self.result_archive_dir, exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                tar_path = os.path.join(self.result_archive_dir, f"archive_{timestamp}.tar.gz")
                with tarfile.open(tar_path, "w:gz") as tar:
                    for file in archive_paths:
                        # Maintain folder structure relative to result_backup_dir
                        arcname = file.relative_to(self.result_backup_dir)
                        tar.add(file, arcname=arcname)
                logger.info(f"Created archive: {tar_path} with files: {archive_paths}")
                await self.cmf_logging_queue.put(tar_path)
            await asyncio.sleep(self.result_archiving_interval)  # e.g., 3 seconds

    def clear_folder(self):
        dvc_path=".dvc"
        cmf_artifacts="cmf_artifacts"
        if os.path.exists(dvc_path):
            
            shutil.rmtree(dvc_path)
            #print(f"Removed {dvc_path}")
            logger.info(f"Removed {dvc_path}")
        if os.path.exists(cmf_artifacts):
            shutil.rmtree(cmf_artifacts)
            #print(f"Removed {cmf_artifacts}")
            logger.info(f"Removed {cmf_artifacts}")
        if os.path.exists(self.pipeline_name):
            os.remove(self.pipeline_name)
            #print(f"Removed {pipeline_name}")
            logger.info(f"Removed {self.pipeline_name}")
        if os.path.exists(self.model_path+".dvc"):
            os.remove(self.model_path+".dvc")
            #print(f"Removed {model_path}.dvc")
            logger.info(f"Removed {self.model_path}.dvc")
    
    async def cmf_archive(self):
        """Archive files from .dvc/cache, cmf_artifacts, and other sources, and add them to the upload queue."""
        #os.makedirs(self.cmf_archive_path, exist_ok=True)
        while not (self.stop_event and self.stop_event.is_set()):
            archive_paths = []
            while not self.cmf_archive_queue.empty():
                file_path = await self.cmf_archive_queue.get()
                archive_paths.append(file_path)
                self.cmf_archive_queue.task_done()
            if archive_paths:
                dvc_path = ".dvc/cache"
                cmf_artifacts = "cmf_artifacts"
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                tar_path = f"cmf_archive_{timestamp}.tar.gz"
                with tarfile.open(tar_path, "w:gz") as tar:
                    if os.path.exists(dvc_path):
                        tar.add(dvc_path, arcname=".dvc/cache")
                    if os.path.exists(cmf_artifacts):
                        tar.add(cmf_artifacts, arcname="cmf_artifacts")
                    if os.path.exists(self.pipeline_file):
                        tar.add(self.pipeline_file, arcname=self.pipeline_file)
                    for file_path in archive_paths:
                        if os.path.exists(file_path):
                            tar.add(file_path, arcname=os.path.basename(file_path))
                await self.cmf_upload_queue.put(tar_path)
            await asyncio.sleep(0.05)  # short sleep for responsiveness

    async def log_result_archives(self):
        while True:
            file_paths = []
            while not self.cmf_logging_queue.empty():
                logger.info("Processing CMF logging queue...")
                file_path = await self.cmf_logging_queue.get()
                file_paths.append(file_path)
                self.cmf_logging_queue.task_done()
            if file_paths:
                self.clear_folder()
                self.cmf_init()
                try:
                    metawriter = cmf.Cmf(self.pipeline_file, self.pipeline_name)
                    stage_name = "inference"
                    execution_name = "inferencing_general"
                    logger.info(f"Logging CMF for pipeline: {self.pipeline_name}, stage: {stage_name}")
                    _ = metawriter.create_context(pipeline_stage=str(stage_name))
                    logger.info(f"Creating context for execution: {execution_name}")
                    _ = metawriter.create_execution(execution_type=str(execution_name))
                    logger.info(f"Logging model: {self.model_path}")
                    _ = metawriter.log_model(self.model_path, event="input")
                    for file_path in file_paths:
                        logger.info(f"Logging dataset: {file_path}")
                        _ = metawriter.log_dataset(file_path, event="output")
                    logger.info("CMF logging completed successfully.")
                except Exception as e:
                    print(f"Error during CMF logging: {e}")
                for file_path in file_paths:
                    file_dvc_path = file_path + ".dvc"
                    await self.cmf_archive_queue.put(file_dvc_path)
                    await self.cmf_archive_queue.put(file_path)
                model_dvc_path = self.model_path + ".dvc"
                await self.cmf_archive_queue.put(self.model_path)
                await self.cmf_archive_queue.put(model_dvc_path)
            await asyncio.sleep(0.05)  # short sleep for responsiveness

    async def cmf_upload(self):
       
        while True:
            upload_paths = []
            while not self.cmf_upload_queue.empty():
                upload_path = await self.cmf_upload_queue.get()
                upload_paths.append(upload_path)
                self.cmf_upload_queue.task_done()
            for cmf_upload_path in upload_paths:
                try:
                    with Plugin() as plugin:
                        plugin.upload_file(cmf_upload_path, timestamp=get_timestamp())
                except Exception as e:
                    print(f"waggle upload failed for {cmf_upload_path}: {e}")
            await asyncio.sleep(0.05)  # short sleep for responsiveness

    async def monitor_and_archive(self):
        async def monitor_file():
            last_hash = None
            
            os.makedirs(self.result_backup_dir, exist_ok=True)
            while True:
                #logging.info(f"Monitoring file: {self.result_path}")
                if os.path.exists(self.result_path):
                    #logging.info(f"File exists: {self.result_path}")
                    with open(self.result_path, "rb") as f:
                        file_bytes = f.read()
                        file_hash = hashlib.md5(file_bytes).hexdigest()
                    if file_hash != last_hash:
                        logging.info(f"File changed: {self.result_path}")
                        last_hash = file_hash
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                        dst_path = os.path.join(self.result_backup_dir, f"{timestamp}_{os.path.basename(self.result_path)}")
                        with open(dst_path, "wb") as f:
                            f.write(file_bytes)
                        #print(f"Copied {self.result_path} to {dst_path}")
                        logging.info(f"Copied {self.result_path} to {dst_path}")
                        if self.archiving == True:
                            logging.info(f"Putting {dst_path} into result_archiving_queue")
                            await self.result_archiving_queue.put(dst_path)
                            #print("result_queue",self.result_archiving_queue)
                        else:
                            await self.cmf_logging_queue.put(dst_path)
                await asyncio.sleep(self.monitoring_interval)

        async def monitor_dir():
            #check if there is new files in the directory
            seen_files = set()
            while True:
                current_files = set(Path(self.result_path).glob("*"))
                new_files = current_files - seen_files
                if new_files:
                    for file_path in new_files:
                        logger.info(f"New file detected: {file_path}")
                        seen_files.add(file_path)
                        if self.archiving:
                            await self.result_archiving_queue.put(str(file_path))
                        else:
                            await self.cmf_logging_queue.put(str(file_path))
                await asyncio.sleep(self.monitoring_interval)
        
        tasks = []
        if self.result_type == "file":
            logging.info(f"Monitoring file: {self.result_path}")
            tasks.append(asyncio.create_task(monitor_file()))
        if self.result_type == "dir":
            tasks.append(asyncio.create_task(monitor_dir()))
        await asyncio.gather(*tasks)
    
    
    def cmf_init(self):
        '''This function initializes the CMF environment in every cmf logging window'''
        '''After removing previous dvc cache and cmf sqlite,it creates new dvc cache and sqlite'''
        #print(f"cmf_init: {self.git_remote_url}")
        logger.info(f"cmf_init: {self.git_remote_url}")
        subprocess.run([
            "cmf", "init", "local",
            "--path", ".",
            "--cmf-server-url", "http://127.0.0.1:80",
            "--git-remote-url", self.git_remote_url
        ])
    

    def schedule(self):
        '''if we are archiving inferencing result, we need to create the result archiving path'''
        '''if there is no archiving of the result, we will simply do the cmf logging on the inference results'''
        '''we will set default window to 4 seconds if otherwise not specified'''
        
        if self.archiving:
            self.result_archiving_path= "result_archiving"
            os.makedirs(self.result_archiving_path, exist_ok=True)
            

    def start(self):
        def run_cmf_logging_in_thread():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self.setup())
        thread= threading.Thread(target=run_cmf_logging_in_thread,daemon=True)
       # thread = threading.Thread(target=lambda: asyncio.run(self.setup()), daemon=True)
        thread.start()
        logging.info("CMF Sage started")
        return thread

    async def log_heartbeat(self):
        while True:
            logger.info("cmfsage is still running...")
            await asyncio.sleep(3)  # log every 3 seconds



if __name__ == "__main__":
    #We define cmf_sage watchdog to keep track of the inference result
    #User needs to provide: 
    '''Pipeline name: What's the pipeline to trace'''
    '''Pipeline file: The mlmd file name, mlmd file be default'''
    '''Model path: The location to the model'''
    '''Result path: The location to the inference result'''
    '''Git remote url: The git remote url to the AI repository'''
    params = {
        "pipeline_name": "wildfire-classification",
        "pipeline_file": "mlmd",
        "model_path": "model.tflite",
        #this can be a directory or a file
        #if this is a file, we will use timestamp to track the files changed. Apply higher monitoring frequency
        #if this is a directory, we will keep track of the files changed in the directory. Apply lower monitoring frequency
        "result_path": "image.jpg",
        "git_remote_url": "https://github.com/hpeliuhan/cmf_proxy_demo.git",
        "archiving": True,
        "logging_interval": 4
    }

    cmf_logger = cmfsage(**params)
    cmf_logger.start()
