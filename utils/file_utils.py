import os
import csv
from typing import List
from loguru import logger


class FileUtils:
    @staticmethod
    def get_all_files_and_folders(directory: str, only_folders: bool = False):
        """
        Get all files and folders in the directory.
        Args:
            directory: Path to the directory.
            only_folders: If only folders should be returned.

        Returns:
            List of folders and optionally files that are in the directory.
        """
        if only_folders:
            return [f for f in os.listdir(directory) if os.path.isdir(os.path.join(directory, f))]
        else:
            return os.listdir(directory)

    @staticmethod
    def get_highest_id_from_csv(file_path: str) -> int:
        """
        Get the highest id from the csv file.
        Args:
            file_path: Path to the file.

        Returns:
            Highest id from the file.
        """
        highest_id = 0
        if os.path.exists(file_path):
            with open(file_path, 'r', newline='', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    highest_id = max(highest_id, int(row['id']))
        return highest_id

    @staticmethod
    def select_folders_datasets(all_folders: List[str], predifined_names: List[str] | None = None) -> List[str]:
        """
        Based on users input, selects folders. If 'not_using' is in the list, it is removed.
        Args:
            all_folders: List of all folders.
            predifined_names: List of predefined names. If chosen, user input won't be asked.

        Returns:
            List of selected folders.
        """
        selected_folders = all_folders
        if 'not_using' in selected_folders:
            selected_folders.remove('not_using')
        folders_str = ', '.join([f'{idx}: {folder}' for idx, folder in enumerate(selected_folders)])
        selected_folder_ids = []
        if predifined_names:
            selected_folder_ids = [selected_folders.index(folder) for folder in selected_folders if folder in predifined_names]
        if not predifined_names or (predifined_names and len(selected_folder_ids) == 0):
            selected_folder_ids = input(f'Select which folders to still do. {folders_str}. Your answer should be list '
                                        f'of numbers corresponding to folders separated by commas. e.g. 0,1,2')
            try:
                selected_folder_ids = [int(folder_id) for folder_id in selected_folder_ids.split(',')]
            except Exception as e:
                logger.error(f'{e} occurred. Using all folders.')
        selected_folders = [selected_folders[folder_id] for folder_id in selected_folder_ids]
        return selected_folders
