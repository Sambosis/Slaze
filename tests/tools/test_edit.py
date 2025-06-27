import pytest
from pathlib import Path
from tools.edit import EditTool

# TODO: Consider adding if __name__ == "__main__": pytest.main() for standalone execution.

@pytest.fixture
def edit_tool():
    """Fixture to create an EditTool instance."""
    return EditTool()

@pytest.mark.asyncio
async def test_view_non_existent_path(edit_tool: EditTool, tmp_path: Path):
    """Test viewing a path that does not exist."""
    non_existent_path = tmp_path / "does_not_exist_dir" / "non_existent_file.txt"
    result = await edit_tool.view(path=non_existent_path)

    assert result.error is not None, "Error should be populated for non-existent path"
    assert "Path not found" in result.error, f"Error message should indicate path not found, but was: {result.error}"
    assert result.output == "", "Output should be empty for non-existent path"

@pytest.mark.asyncio
async def test_view_directory(edit_tool: EditTool, tmp_path: Path):
    """Test viewing a directory."""
    test_dir = tmp_path / "test_view_dir"
    test_dir.mkdir()
    file1 = test_dir / "file1.txt"
    file1.write_text("content1")
    subdir = test_dir / "subdir"
    subdir.mkdir()
    file2 = subdir / "file2.txt"
    file2.write_text("content2")
    hidden_file = test_dir / ".hidden.txt"
    hidden_file.write_text("hidden_content")
    hidden_subdir = subdir / ".hidden_subdir"
    hidden_subdir.mkdir()

    result = await edit_tool.view(path=test_dir)

    assert result.error is None, f"Error should be None for a valid directory, but was: {result.error}"
    assert result.output is not None, "Output should not be None for a valid directory"

    # Check for expected file and directory names
    # The paths in output are absolute, so we need to check against resolved paths
    assert str(file1.resolve()) in result.output, f"Expected {file1.resolve()} in output:\n{result.output}"
    assert str(subdir.resolve()) in result.output, f"Expected {subdir.resolve()} in output:\n{result.output}"
    assert str(file2.resolve()) in result.output, f"Expected {file2.resolve()} in output (recursive listing expected by default up to 2 levels):\n{result.output}"

    # Check that hidden files/dirs are not listed
    assert str(hidden_file.resolve()) not in result.output, f"Hidden file {hidden_file.resolve()} should not be in output:\n{result.output}"
    assert str(hidden_subdir.resolve()) not in result.output, f"Hidden subdir {hidden_subdir.resolve()} should not be in output:\n{result.output}"

    assert f"Here's the files and directories up to 2 levels deep in {test_dir}" in result.output


@pytest.mark.asyncio
async def test_view_file_full(edit_tool: EditTool, tmp_path: Path):
    """Test viewing a file's full content."""
    test_file = tmp_path / "test_view_file_full.txt"
    file_lines = ["Line 1: Hello World", "Line 2: Testing view", "Line 3: End of file."]
    file_content = "\n".join(file_lines)
    test_file.write_text(file_content)

    result = await edit_tool.view(path=test_file)

    assert result.error is None, f"Error should be None, but was: {result.error}"
    assert result.output is not None

    expected_output = edit_tool._make_output(file_content, str(test_file))
    assert result.output == expected_output, f"Output did not match expected.\nExpected:\n{expected_output}\nGot:\n{result.output}"

@pytest.mark.asyncio
@pytest.mark.parametrize("view_range, expected_lines_indices", [
    ([1, 3], [0, 1, 2]),      # All lines (assuming 3 lines in test file)
    ([2, 2], [1]),            # Line 2 only
    ([1, 2], [0, 1]),         # Lines 1 to 2
    ([3, -1], [2]),           # Line 3 to end (assuming 3 lines total)
    ([1, -1], [0, 1, 2]),      # All lines using -1
])
async def test_view_file_with_range(edit_tool: EditTool, tmp_path: Path, view_range: list[int], expected_lines_indices: list[int]):
    """Test viewing a file with various valid view_ranges."""
    test_file = tmp_path / "test_view_file_range.txt"
    base_file_lines = ["Line 1: Alpha", "Line 2: Beta", "Line 3: Gamma"]
    test_file.write_text("\n".join(base_file_lines))

    # Determine expected content based on range
    start_idx = view_range[0] - 1
    if view_range[1] == -1:
        end_idx = len(base_file_lines)
    else:
        end_idx = view_range[1]

    expected_content_lines = base_file_lines[start_idx:end_idx]
    expected_content = "\n".join(expected_content_lines)

    result = await edit_tool.view(path=test_file, view_range=view_range)

    assert result.error is None, f"Error should be None for range {view_range}, but was: {result.error}"
    assert result.output is not None

    # The _make_output method adds line numbers based on the original file's line numbers.
    # So, if view_range is [2,2], the output line will be "     2   Line 2: Beta"
    expected_output = edit_tool._make_output(expected_content, str(test_file), init_line=view_range[0])
    assert result.output == expected_output, f"Output mismatch for range {view_range}.\nExpected:\n{expected_output}\nGot:\n{result.output}"


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid_range, error_message_part", [
    ([0, 2], "Its first element `0` should be within the range of lines"), # Line numbers are 1-indexed
    ([10, 12], "Its first element `10` should be within the range of lines"), # Start line out of bounds (assuming 3 lines in test file)
    ([1, 0], "Its second element `0` should be larger or equal than its first `1`"),     # End line before start line
    ([1, 100], "Its second element `100` should be smaller than the number of lines"),# End line out of bounds
    ([2, 1], "Its second element `1` should be larger or equal than its first `2`"), # End before start
])
async def test_view_file_invalid_range(edit_tool: EditTool, tmp_path: Path, invalid_range: list[int], error_message_part: str):
    """Test viewing a file with various invalid view_ranges."""
    test_file = tmp_path / "test_view_file_invalid_range.txt"
    # Create a file with 3 lines for these tests
    test_file.write_text("Line 1\nLine 2\nLine 3")

    result = await edit_tool.view(path=test_file, view_range=invalid_range)

    assert result.error is not None, f"Error should be populated for invalid range {invalid_range}, but was None. Output: {result.output}"
    assert error_message_part in result.error, f"Error message for range {invalid_range} did not contain '{error_message_part}'. Got: '{result.error}'"
    # output can sometimes be populated by the __call__ method's formatting even on error, so don't assert result.output == ""

@pytest.mark.asyncio
async def test_view_empty_file(edit_tool: EditTool, tmp_path: Path):
    """Test viewing an empty file."""
    test_file = tmp_path / "empty_file.txt"
    test_file.write_text("")

    result = await edit_tool.view(path=test_file)
    assert result.error is None, f"Error should be None for empty file, but was: {result.error}"
    expected_output = edit_tool._make_output("", str(test_file))
    assert result.output == expected_output, f"Output mismatch for empty file.\nExpected:\n{expected_output}\nGot:\n{result.output}"

@pytest.mark.asyncio
async def test_view_file_with_range_on_empty_file(edit_tool: EditTool, tmp_path: Path):
    """Test viewing an empty file with a view_range."""
    test_file = tmp_path / "empty_file_for_range.txt"
    test_file.write_text("")

    # view_range [1,1] on an empty file. read_file returns "", splitlines is [''] so n_lines_file is 1.
    # init_line=1, final_line=1. file_lines[0:1] is ['']
    # This is a bit of an edge case. The current code allows this.
    # The first element '1' is not > n_lines_file (1)
    # The second element '1' is not > n_lines_file (1)
    result_one_one = await edit_tool.view(path=test_file, view_range=[1,1])
    assert result_one_one.error is None, f"Error for view_range [1,1] on empty file: {result_one_one.error}"
    expected_output_one_one = edit_tool._make_output("", str(test_file), init_line=1) # _make_output with "" content
    assert result_one_one.output == expected_output_one_one

    # view_range [1,-1] on an empty file
    result_one_neg_one = await edit_tool.view(path=test_file, view_range=[1,-1])
    assert result_one_neg_one.error is None, f"Error for view_range [1,-1] on empty file: {result_one_neg_one.error}"
    expected_output_one_neg_one = edit_tool._make_output("", str(test_file), init_line=1)
    assert result_one_neg_one.output == expected_output_one_neg_one

    # view_range like [2,2] should fail as init_line > n_lines_file
    result_invalid = await edit_tool.view(path=test_file, view_range=[2,2])
    assert result_invalid.error is not None, "Error should be populated for out-of-bounds range on empty file"
    expected_error_msg = "Invalid `view_range`: [2, 2]. Its first element `2` should be within the range of lines of the file: [1, 1]"
    assert expected_error_msg in result_invalid.error, f"Expected '{expected_error_msg}' to be in '{result_invalid.error}'"
