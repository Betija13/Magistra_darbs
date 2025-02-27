import os
import csv
from typing import List
from loguru import logger

class FileUtils:
    @staticmethod
    def get_all_files_and_folders(directory: str, only_folders: bool = False):
        if only_folders:
            return [f for f in os.listdir(directory) if os.path.isdir(os.path.join(directory, f))]
        else:
            return os.listdir(directory)

    @staticmethod
    def get_result_files(directory: str):
        return [f for f in os.listdir(directory) if f.startswith('result_') or f.startswith('results_')]

    @staticmethod
    def get_highest_id_from_csv(file_path: str) -> int:
        highest_id = 0
        if os.path.exists(file_path):
            with open(file_path, 'r', newline='', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    highest_id = max(highest_id, int(row['id']))
        return highest_id

    @staticmethod
    def select_folders_datasets(all_folders: List[str]) -> List[str]:
        selected_folders = all_folders
        selected_folders.remove('not_using')
        folders_str = ', '.join([f'{idx}: {folder}' for idx, folder in enumerate(selected_folders)])
        selected_folder_ids = input(f'Select which folders to still do. {folders_str}. Your answer should be list of numbers corresponding to folders separated by commas. e.g. 0,1,2')
        try:
            selected_folder_ids = [int(folder_id) for folder_id in selected_folder_ids.split(',')]
            selected_folders = [selected_folders[folder_id] for folder_id in selected_folder_ids]
        except Exception as e:
            logger.error(f'{e} occurred. Using all folders.')
        return selected_folders
