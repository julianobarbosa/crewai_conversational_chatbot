import subprocess

from crewai.tools import BaseTool


class PythonLintTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="Python Lint Tool",
            description="Checks Python code for style and formatting issues using pylint or flake8.",
        )

    # Implement the abstract method `check_tool` for check if `pylint` is installed.
    def check_tool(self):
        """
        Check if the linting tool is installed on the system.

        :return: True if the tool is installed, False otherwise.
        """
        try:
            # implement logging logic
            self.logger.info("Checking if pylint is installed...")

            # Run `pylint` with the `--version` flag to check if it is installed
            result = subprocess.run(
                ["pylint", "--version"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            # Check if the linting tool is installed
            if result.returncode == 0:
                return True
            else:
                return False

        except FileNotFoundError:
            return False

    # implement the method to install the tool if it is not installed.
    def install_tool(self):
        """
        Install the linting tool using pip.

        :return: True if the tool was installed successfully, False otherwise.
        """
        try:
            # implement logging logic
            self.logger.info("Installing pylint using pip...")

            # check if the tool is installed, if not, install it using pip.
            if self.check_tool():
                return True

            # Install `pylint` using pip
            result = subprocess.run(
                ["pip", "install", "pylint"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            # Check if the installation was successful
            if result.returncode == 0:
                return True
            else:
                return False

        except Exception as e:
            return False

    def execute(self, file_path):
        """
        Run the linting tool on the specified Python file and return the results.

        :param file_path: Path to the Python file to be linted.
        :return: Linting output as a string.
        """
        try:
            # implement logging logic
            self.logger.info("Running pylint on the specified file...")

            # Run `pylint` on the specified file
            result = subprocess.run(
                ["pylint", file_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            # Check if the linting command was successful
            if result.returncode == 0:
                return f"Linting Successful: No issues found in {file_path}\n{result.stdout}"
            else:
                return f"Linting Report for {file_path}:\n{result.stdout}\nErrors:\n{result.stderr}"

        except FileNotFoundError:
            return "Error: pylint is not installed. Please install it using `pip install pylint`."
        except Exception as e:
            return f"An unexpected error occurred: {e}"


# Example usage
if __name__ == "__main__":
    # implement logging logic
    self.logger.info("Running the Python Lint Tool...")

    lint_tool = PythonLintTool()
    # Replace 'example.py' with the path to your Python file
    report = lint_tool.execute("example.py")
    print(report)
