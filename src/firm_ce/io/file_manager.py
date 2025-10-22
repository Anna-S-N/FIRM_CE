import csv
import os
from pathlib import Path
from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray


class ImportCSV:
    """
    Class for importing CSV files from a given repository.

    This class is designed to handle model configuration files such as
    scenarios, generators, fuels, lines, storages, and others. Each CSV
    is loaded into a nested dictionary indexed by the `id` column.
    """

    def __init__(self, repository: Path) -> None:
        """
        Initialize the importer with a path to the repository.

        Parameters:
        -------
        repository (Path): Path to the directory containing CSV files.

        Raises
        ------
        FileNotFoundError: If the specified repository path does not exist.
        """

        self.repository = Path(repository)
        
        self.config_filenames = (
            "scenarios",
            "generators",
            "fuels",
            "lines",
            "storages",
            "config",
            "initial_guess",
            "datafiles",
        )

        if not self.repository.is_dir():
            raise FileNotFoundError(f"Repository {repository} does not exist.")

    def get_data(self, filename: str) -> Dict[str, Dict[str, Any]]:
        """
        Load a CSV file into a nested dictionary keyed by 'id'.

        Parameters:
        -------
        filename (str): Name of the CSV file to load, including the extension.

        Returns:
        -------
        Dict[str, Dict[str, Any]]: Dictionary of records keyed by 'id'.

        Raises:
        -------
        FileNotFoundError: If the specified repository path does not exist.
        """
        
        filepath = self.repository.joinpath(filename)
        if not filepath.is_file():
            raise FileNotFoundError(f"File {filepath} does not exist.")
        imported_dict = pd.read_csv(filepath, index_col="id").to_dict(orient="index")
        for idx in imported_dict:
            imported_dict[idx]["id"] = idx
        return imported_dict

    def get_config_dict(self) -> Dict[str, Dict[str, Any]]:
        """
        Load all predefined configuration CSVs into a dictionary.

        Returns
        -------
        Dict[str, Dict[str, Any]]: A dictionary mapping each configuration filename
        (without extension) to its corresponding record dictionary.
        """
        return {fn: self.get_data(fn + ".csv") for fn in self.config_filenames}
        

class ImportDatafile:
    """
    Class for importing a single CSV datafile into a dictionary of NDArrays.
    """

    def __init__(self, repository: Path, filename: str) -> None:
        """
        Initialize the datafile importer.

        Parameters:
        -------
        repository (Path): Directory containing the CSV file.
        filename (str): Name of the CSV datafile to import.

        Raises:
        -------
        FileNotFoundError: If the specified repository path does not exist.
        """

        self.repository = Path(repository)
        self.filename = filename

        if not self.repository.is_dir():
            raise FileNotFoundError(f"Repository {repository} does not exist.")

    def __repr__(self) -> str:
        return f"ImportDatafile ({self.filename!r})"

    def get_data(self) -> Dict[str, NDArray]:
        """
        Load the CSV data into a dictionary of NumPy arrays.

        Returns:
        -------
        Dict[str, NDArray]: Dictionary where keys are column names and values are NumPy arrays.

        Raises:
        -------
        FileNotFoundError: If the specified repository path does not exist.
        """
        filepath = self.repository.joinpath(self.filename)
        if not filepath.is_file():
            raise FileNotFoundError(f"File {filepath} does not exist.")

        df = pd.read_csv(filepath)
        return {col: df[col].to_numpy() for col in df.columns}


class DataFile:
    """
    Container for a named datafile and its content.
    """

    def __init__(self, filename: str, datafile_type: str, file_directory: str) -> None:
        """
        Initialize the DataFile with a filename and its type.

        Parameters:
        -------
        filename (str): Name of the datafile.
        datafile_type (str): Descriptive type of the datafile (e.g., 'time series').
        file_directory (str): Path to the directory containing the datafile.
        """
        self.name = filename
        self.type = datafile_type
        self.file_directory = file_directory
        self.data = ImportDatafile(file_directory, filename).get_data()

    def __repr__(self) -> str:
        return f"DataFile ({self.name!r}, {self.type!r}, {self.data!r})"


class ResultFile:
    """
    Container class for saving model results into a CSV file.

    Handles writing headers, formatting data, and rounding values
    before output.
    """

    def __init__(
        self,
        file_type: str,
        target_directory: str,
        header: List[str],
        data_array: Union[NDArray[np.float64], NDArray[np.int64]],
        decimals: Union[int, None] = None,
    ):
        """
        Initialise a result file object.

        Parameters:
        ----------
        file_type (str): Identifier for the file (used as the filename base).
        target_directory (str): Directory where the file will be saved.
        header (List[str]): List of column headers for the CSV file.
        data_array (Union[NDArray[np.float64], NDArray[np.int64]]): Data to
            write into the CSV file.
        decimals (Union[int, None], optional): Number of decimal places to round
            data values to. If None, values are written without rounding.
        """
        self.name = file_type + ".csv"
        self.type = file_type
        self.target_directory = target_directory
        self.header = header
        self.data = data_array
        self.decimals = decimals

    def __repr__(self) -> str:
        return f"ResultFile ({self.type!r})"

    def write(self):
        """
        Write the data to a CSV file.

        Multi-line headers are possible.

        Each row of data_array is written to the file. Optionally,
        values are rounded to the specified number of decimals.

        Returns:
        -------
        None.

        Side-effects:
        ------------
        Creates a CSV file in target_directory with the name
        <file_type>.csv and prints a confirmation message.
        """
        with open(os.path.join(self.target_directory, self.name), mode="w", newline="") as file:
            writer = csv.writer(file)
            if self.header is not None:
                for row in self.header:
                    writer.writerow(row)
            for row in self.data:
                writer.writerow(np.round(row, decimals=self.decimals) if self.decimals is not None else row)
        print(f"Saved {self.name} to {self.target_directory}")
        return None


def import_config_csvs(config_directory: str) -> Dict[str, Any]:
    """
    Load all model configuration CSVs into a single dictionary.

    Parameters:
    ----------
    config_directory (str): Path to the directory containing configuration CSV files.

    Returns:
    -------
    Dict[str, Any]: A dictionary mapping configuration filenames (without extension)
        to their respective record dictionaries.
    """

    csv_importer = ImportCSV(config_directory)
    data = csv_importer.get_config_dict()

    return data
