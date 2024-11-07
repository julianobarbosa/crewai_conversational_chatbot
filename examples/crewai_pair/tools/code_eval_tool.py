import subprocess

from crewai.tools import BaseTool


class CodeEvaluationTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="Code Evaluation Tool",
            description="Executes and evaluates Python code.",
        )

    def execute(self, file_path):
        """
        Run the specified Python file and capture the output or errors.

        :param file_path: Path to the Python file to be evaluated.
        :return: Output of the code execution or error messages.
        """
        try:
            # Run the Python file using subprocess
            result = subprocess.run(
                ["python", file_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            if result.returncode == 0:
                return f"Execution Successful:\n{result.stdout}"
            else:
                return f"Execution Errors:\n{result.stderr}"

        except FileNotFoundError:
            return "Error: Python interpreter not found or file does not exist."
        except Exception as e:
            return f"An unexpected error occurred during code evaluation: {e}"


# Example usage
if __name__ == "__main__":
    code_eval_tool = CodeEvaluationTool()
    # Replace 'example.py' with the path to your Python file
    result = code_eval_tool.execute("example.py")
