import re
import sys
import os
import argparse

from typing import List, Optional, Tuple

# Enum pattern:
#   ^                                - Matches the start of the line
#   (\s*)                            - Captures the indentation before the enum definition (stored in group 1)
#   class\s+                         - Matches the class keyword followed by spaces
#   (\w+)                            - Captures the enum name (stored in group 2)
#   \s*\(\s*(?:enum\.)?Enum\s*\)\s*: - Matches the Enum base class
ENUM_PATTERN = re.compile(r'^(\s*)class\s+(\w+)\s*\(\s*(?:enum\.)?Enum\s*\)\s*:')

# Enum value pattern:
#   \s+      - Matches one or more spaces
#   (\w+)    - Captures the enum value name (stored in group 1)
#   \s*=     - Matches the equal sign with
ENUM_VALUE_PATTERN = re.compile(r'\s+(\w+)\s*=')

# Class pattern:
#   ^               - Matches the start of the line
#   (\s*)           - Captures the indentation before the class definition (stored in group 1)
#   cdef\s+class\s+ - Matches the class keyword followed by spaces
#   ([\w\[\]\(\)]+) - Captures the class name (stored in group 2)
#   \s*:            - Matches the class definition
CLASS_PATTERN = re.compile(r'^(\s*)cdef\s+class\s+([\w\[\]\(\)]+)\s*:')

# Class property pattern:
#   \s+@property\s*      - Matches the @property decorator
#   \s+@(\w+)\.setter\s* - Captures the name of the setter (stored in group 1)
GETTER_PATTERN = re.compile(r'\s+@property\s*')
SETTER_PATTERN = re.compile(r'\s+@(\w+)\.setter\s*')

# Static method pattern:
#   \s+@staticmethod\s* - Matches the @staticmethod decorator
STATIC_METHOD_PATTERN = re.compile(r'\s+@staticmethod\s*')

# Function pattern:
#   ^               - Matches the start of the line
#   (\s*)           - Captures the indentation before the function definition (stored in group 1)
#   def\s+(\w+)\(   - Captures the function name (stored in group 2)
FUNCTION_PATTERN = re.compile(r'^(\s*)def\s+(\w+)\(')

# Function pattern when it is a oneliner, with return type catched:
#   ^               - Matches the start of the line
#   (\s*)           - Captures the indentation before the function definition (stored in group 1)
#   def\s+(\w+)\(   - Captures the function name (stored in group 2)
#   .*->\s*         - Matches the return type arrow
#   ([\w\[\]\(\)\.]+) - Captures the return type (stored in group 3)
ONELINE_FUNCTION_PATTERN = re.compile(r'^(\s*)def\s+(\w+)\(.*->\s*([\w\[\]\(\)\.]+)\s*:')

# Cython Docstring start pattern
#   ^      - Matches the start of the line
#   (\s*)  - Captures the indentation (stored in group 1)
#   ##     - Matches ##
#   \s*$   - Optional trailing whitespace to end of line
CYTHON_DOCSTRING_START_PATTERN = re.compile(r'^(\s*)##\s*$')

# Cython Docstring inner pattern
#   ^      - Matches the start of the line
#   (\s*)  - Captures the indentation (stored in group 1)
#   #      - Matches #
#   (.*)$  - Captures everything to end of line (stored in group 2)
CYTHON_DOCSTRING_INNER_PATTERN = re.compile(r'^(\s*)#(.*)$')

# Methods to ignore
METHODS_TO_COMPLETELY_IGNORE = set(["__cinit__"])


def remove_static_type_arg(argument: str) -> str:
    """
    Remove the static type from a Cython argument

    Passed argument is supposed to be extracted from a valid Cython function signature.

    E.g.:
        - "unsigned int i" -> "i"
        - "int i" -> "i"

    Args:
        argument: The argument to clean

    Returns:
        The cleant argument
    """
    argument = argument.strip()
    if ":" in argument or argument == "":
        return argument

    if "=" in argument:
        argument, right_side = argument.split("=")
        return f"{argument.split()[-1]} = {right_side}"

    return argument.split()[-1]


def extract_setter_type(setter_line: str) -> Optional[str]:
    """
    Extract the type of the setter parameter from a Cython setter function signature.

    Handles both Python-style annotations (value: Type) and Cython-style (Type value).

    E.g.:
        - "def input(self, InputType input_t):" -> "InputType"
        - "def value(self, value: float):" -> "float"
        - "def input(self, input_t):" -> None (no type)

    Args:
        setter_line: The line containing the setter function definition.

    Returns:
        The type of the setter parameter, or None if no type is found.
    """
    # Find the content inside the parentheses
    if "(" not in setter_line or ")" not in setter_line:
        return None

    params_start = setter_line.find("(")
    params_end = setter_line.rfind(")")
    params = setter_line[params_start + 1:params_end]

    # Split by comma and skip 'self' parameter
    param_list = [p.strip() for p in params.split(",")]
    if len(param_list) < 2:
        return None

    # The second parameter is the value parameter
    value_param = param_list[1].strip()

    # Check for Python-style annotation (name: Type)
    if ":" in value_param:
        parts = value_param.split(":")
        if len(parts) >= 2:
            return parts[1].strip()

    # Check for Cython-style (Type name)
    parts = value_param.split()
    if len(parts) >= 2:
        # The type is all parts except the last one (which is the name)
        return " ".join(parts[:-1])

    return None


def get_end_of_class_pattern(class_indent: int) -> re.Pattern:
    """
    Compile the pattern to match the end of a class definition

    Args:
        class_indent: The indentation before the class definition

    Returns:
        The compiled pattern
    """
    if class_indent <= 0:
        # End of class pattern:
        #   ^\S  - Matches a line starting with something other than a space
        end_of_class_pattern = re.compile(r'^\S')
    else:
        # End of class pattern:
        #   ^\s{,class_indent-1}\S  - Matches a line that starts with less indentation than the class definition
        end_of_class_pattern = re.compile('^\\s{,' + str(class_indent) + '}\\S')
    return end_of_class_pattern


def get_stubed_body(ret_signature: str) -> str:
    """
    Get the stubed body of a function given its return signature

    Args:
        ret_signature: The return signature of the function

    Returns:
        The stubed body of the function
    """
    ret_signature_lower = ret_signature.strip().lower()
    if ret_signature_lower == "none":
        return "pass"
    if ret_signature_lower == "any":
        return "return None"
    if ret_signature_lower == "list":
        return "return []"
    if ret_signature_lower == "tuple":
        return "return ()"
    if ret_signature_lower == "dict":
        return "return {}"
    if ret_signature_lower.startswith("optional"):
        return "return None"
    if ret_signature_lower.startswith("union"):
        return "return None"
    if ret_signature_lower.startswith("numpy.ndarray"):
        return "return np.array([])"

    return f"return {ret_signature}()"


def extract_docstring(pyx_content: List[str], docstring_idx: int) -> Tuple[int, List[str]]:
    """
    Extract the docstring.

    Args:
        pyx_content: The content of the .pyx file split by lines
        docstring_idx: The index of the line where the docstring starts

    Returns:
        The index of the line after the docstring
        The extracted docstring, if any, as a list of lines
    """
    if docstring_idx >= len(pyx_content):
        return docstring_idx, []

    if (match := CYTHON_DOCSTRING_START_PATTERN.match(pyx_content[docstring_idx])):
        indent = match.group(1)
        docstring = []
        docstring_idx += 1
        while docstring_idx < len(pyx_content):
            if (inner_match := CYTHON_DOCSTRING_INNER_PATTERN.match(pyx_content[docstring_idx])):
                inner_indent = inner_match.group(1)
                if len(inner_indent) != len(indent):
                    break
                docstring.append(inner_match.group(2).rstrip())
            else:
                break
            docstring_idx += 1
        return docstring_idx, docstring

    # No docstring found
    return docstring_idx, []


def extract_signature_from_function(pyx_content: List[str], fct_idx: int) -> Tuple[int, str, str]:
    """
    Extract a function definition from a Cython file

    Args:
        pyx_content: The content of the .pyx file split by lines
        fct_idx: The index of the line where the function definition starts

    Returns:
        The index of the line after the function definition and the extracted stub
        The arguments signature of the function definition
        The return type signature of the function definition
    """
    arg_signature = pyx_content[fct_idx][pyx_content[fct_idx].find("(") + 1:]
    ret_signature = ""

    # Ensure ( and ) are balanced, could be on multiple lines
    while fct_idx < len(pyx_content) and (arg_signature.count("(") - arg_signature.count(")") > 0):
        fct_idx += 1
        if fct_idx < len(pyx_content):
            arg_signature += pyx_content[fct_idx]
    # Once ( and ) are balanced, continue until finding the `:`
    while fct_idx < len(pyx_content) and not ':' in arg_signature[arg_signature.rfind(")"):]:
        fct_idx += 1
        if fct_idx < len(pyx_content):
            arg_signature += pyx_content[fct_idx]

    arg_signature = arg_signature[:arg_signature.rfind(":")]

    # Return type is specified
    if "->" in arg_signature:
        arg_signature, ret_signature = arg_signature.split("->")
        ret_signature = ret_signature.strip().replace("(", "[").replace(")", "]")
    else:
        ret_signature = "None"

    arg_signature = [remove_static_type_arg(a) for a in arg_signature[:arg_signature.rfind(")")].strip().split(",")]

    return fct_idx + 1, ", ".join(arg_signature), ret_signature


def extract_stub_from_enum(enum_name: str, enum_indent: int, pyx_content: List[str], enum_idx: int, docstring_enum: List[str], missing_docstrings: Optional[List[str]] = None) -> Tuple[int, str]:
    """
    Extract an enum definition from a Cython file

    Args:
        enum_name: The name of the enum
        enum_indent: The indentation before the enum definition
        pyx_content: The content of the .pyx file split by lines
        enum_idx: The index of the line where the enum definition starts
        docstring_enum: The docstring of the enum definition
        missing_docstrings: List to append missing docstrings to

    Returns:
        The index of the line after the enum definition and the extracted stub
        The stub of the enum definition
    """
    end_of_enum_pattern = get_end_of_class_pattern(enum_indent)

    current_docstring = []

    stub = f"class {enum_name}(enum.Enum):\n"
    if len(docstring_enum) > 0:
        stub += " " * 4 + "\"\"\"\n"
        for line_docstring in docstring_enum:
            stub += " " * 4 + line_docstring + "\n"
        stub += " " * 4 + "\"\"\"\n"
    elif missing_docstrings is not None:
        missing_docstrings.append(f"Enum '{enum_name}'")

    enum_idx += 1
    while enum_idx < len(pyx_content):
        # End of enum reached
        if end_of_enum_pattern.match(pyx_content[enum_idx]):
            break

        # Extract docstring if any
        enum_idx, docstring = extract_docstring(pyx_content, enum_idx)
        if len(docstring) > 0:
            current_docstring = format_docstring(docstring)

        # Extract enum value
        elif (match := ENUM_VALUE_PATTERN.match(pyx_content[enum_idx])):
            value_name = match.group(1)
            stub += f"    {value_name} = enum.auto()\n"

        # Extract method
        elif (match := FUNCTION_PATTERN.match(pyx_content[enum_idx])):
            method_name = match.group(2)
            enum_idx, arg_signature, ret_signature = extract_signature_from_function(pyx_content, enum_idx)
            stub += f"    def {method_name}({arg_signature}) -> {ret_signature}:\n"
            if len(current_docstring) > 0:
                stub += "\n" + " " * (enum_indent + 8) + "\"\"\""
                for line_docstring in current_docstring:
                    stub += "\n" + " " * (enum_indent + 8) + line_docstring
                stub += "\n" + " " * (enum_indent + 8) + "\"\"\"\n"
                current_docstring = []
            elif missing_docstrings is not None and not method_name.startswith("_"):
                missing_docstrings.append(f"Method '{enum_name}.{method_name}'")
            stub += f"        {get_stubed_body(ret_signature)}\n\n"

        # Continue looking for other lines
        enum_idx += 1

    return enum_idx, stub


def extract_stub_from_class(class_name: str, class_indent: int, pyx_content: List[str], class_idx: int, docstring_cls : List[str], missing_docstrings: Optional[List[str]] = None) -> Tuple[int, str]:
    """
    Extract a class definition from a Cython file

    Args:
        class_name: The name of the class
        class_indent: The indentation before the class definition
        pyx_content: The content of the .pyx file split by lines
        class_idx: The index of the line where the class definition starts
        docstring_cls: The docstring of the class definition
        missing_docstrings: List to append missing docstrings to

    Returns:
        The index of the line after the class definition and the extracted stub
        The stub of the class definition
    """
    end_of_class_pattern = get_end_of_class_pattern(class_indent)

    stub_methods = ""
    current_docstring = []
    is_next_method_static = False

    getter_docstring = dict()  # {prop_name: docstring}
    getter_types = dict()  # {prop_name: type}
    setters = set()  # {prop_name}
    setter_types = dict()  # {prop_name: type}
    static_methods = set()  # {method_name}

    class_idx += 1
    while class_idx < len(pyx_content):
        # End of class reached
        if end_of_class_pattern.match(pyx_content[class_idx]):
            break

        # Extract docstring if any
        class_idx, docstring = extract_docstring(pyx_content, class_idx)
        if len(docstring) > 0:
            current_docstring = format_docstring(docstring)

        # Extract property getter
        elif (GETTER_PATTERN.match(pyx_content[class_idx])):
            # Method is expected on the following line
            if (match := ONELINE_FUNCTION_PATTERN.match(pyx_content[class_idx + 1])):
                prop_name = match.group(2)
                prop_type = match.group(3)
                class_idx += 2
                if prop_name in getter_types:
                    print(f"Error: In class '{class_name}', found at least two getters for property {prop_name}")
                    sys.exit(1)
                getter_types[prop_name] = prop_type
                getter_docstring[prop_name] = current_docstring
                if len(current_docstring) == 0 and missing_docstrings is not None:
                    missing_docstrings.append(f"Property '{class_name}.{prop_name}'")
                current_docstring = []
            else:
                # Property getter without return type. Should not happen, but better to be safe than sorry.
                if (match := FUNCTION_PATTERN.match(pyx_content[class_idx + 1])):
                    prop_name = match.group(2)
                    print(f"Warning: Property '{class_name}.{prop_name}' has no return type annotation")
                class_idx += 1

        # Extract property setter
        elif (match := SETTER_PATTERN.match(pyx_content[class_idx])):
            prop_name = match.group(1)
            # method is either on the current or the following line
            if (def_match := FUNCTION_PATTERN.match(pyx_content[class_idx + 1])):
                if prop_name == def_match.group(2):
                    # Extract setter type from the signature
                    setter_line = pyx_content[class_idx + 1]
                    setter_type = extract_setter_type(setter_line)
                    if setter_type:
                        setter_types[prop_name] = setter_type
                    class_idx += 2
                    if prop_name in setters:
                        print(f"Error: In class '{class_name}', found at least two setters for property {prop_name}")
                        sys.exit(1)
                    setters.add(prop_name)
                else:
                    class_idx += 1
            else:
                class_idx += 1
            current_docstring = []

        # Detect @staticmethod decorator, only for Python def functions, not cdef
        elif STATIC_METHOD_PATTERN.match(pyx_content[class_idx]):
            # Check if the next line is a 'def' function (not 'cdef')
            next_line = pyx_content[class_idx + 1] if class_idx + 1 < len(pyx_content) else ""
            if FUNCTION_PATTERN.match(next_line):
                is_next_method_static = True
            class_idx += 1

        # Extract method
        elif (match := FUNCTION_PATTERN.match(pyx_content[class_idx])):
            method_name = match.group(2)
            if method_name in METHODS_TO_COMPLETELY_IGNORE:
                class_idx += 1
                is_next_method_static = False
                continue

            class_idx, arg_signature, ret_signature = extract_signature_from_function(pyx_content, class_idx)

            # Check if this is a static method
            if is_next_method_static:
                stub_methods += "    @staticmethod\n"
                static_methods.add(method_name)
                is_next_method_static = False

            stub_methods += f"    def {method_name}({arg_signature}) -> {ret_signature}:\n"
            if len(current_docstring) > 0:
                stub_methods += " " * 8 + "\"\"\"\n"
                for line_docstring in current_docstring:
                    stub_methods += " " * 8 + line_docstring + "\n"
                stub_methods += " " * 8 + "\"\"\"\n"
                current_docstring = []
            elif missing_docstrings is not None and not method_name.startswith("_"):
                missing_docstrings.append(f"Method '{class_name}.{method_name}'")
            stub_methods += f"        {get_stubed_body(ret_signature)}\n\n"

        # Something else (empty line, comment, ...)
        else:
            class_idx += 1

    stub = f"class {class_name}:\n"
    if len(docstring_cls) > 0:
        stub += " " * 4 + "\"\"\"\n"
        for line_docstring in docstring_cls:
            stub += " " * 4 + line_docstring + "\n"
        stub += " " * 4 + "\"\"\"\n"
    elif missing_docstrings is not None:
        missing_docstrings.append(f"Class '{class_name}'")

    stub += "    def __init__(self, *args, **kwargs) -> None: ...\n\n"

    # Add properties to the stub
    for prop_name in setters.union(getter_types.keys()):
        if prop_name in getter_types:
            prop_type = getter_types[prop_name]
            stub +=  "    @property\n"
            stub += f"    def {prop_name}(self) -> {prop_type}:\n"
            if (prop_name in getter_docstring) and (len(getter_docstring[prop_name]) > 0):
                if (prop_name in getter_docstring) and (len(getter_docstring[prop_name]) > 0):
                    stub += " " * 8 + "\"\"\"\n"
                    for line_docstring in getter_docstring[prop_name]:
                        stub += " " * 8 + line_docstring + "\n"
                    stub += " " * 8 + "\"\"\"\n"
            stub += f"        {get_stubed_body(prop_type)}\n\n"
        if prop_name in setters:
            # Determine the setter parameter type
            # Priority: 1) explicit setter type, 2) getter type, 3) Any
            setter_param_type = "Any"
            if prop_name in setter_types and setter_types[prop_name]:
                setter_param_type = setter_types[prop_name]
            elif prop_name in getter_types:
                setter_param_type = getter_types[prop_name]

            stub += f"    @{prop_name}.setter\n"
            stub += f"    def {prop_name}(self, {prop_name}: {setter_param_type}) -> None:\n"
            stub += "        pass\n\n"
    # Add methods to the stub
    stub += stub_methods

    return class_idx, stub


def extract_stub_from_pyx(pyx_file_path: str, output_file_path: Optional[str] = None, check_docstrings: bool = False) -> List[str]:
    """
    Extract Python stub definitions from a .pyx file
    """
    if output_file_path is None:
        # Use same name but with .py extension
        output_file_path = os.path.splitext(pyx_file_path)[0] + '.pyi'

    with open(pyx_file_path, 'r', encoding='utf-8') as f:
        pyx_content = f.read().split("\n")

    # Prepare the stub file with imports
    stub = """from __future__ import annotations
import enum
import numpy as np
from typing import List, Tuple, Dict, Optional, Union, Any, overload, Mapping, MutableMapping
import numpy.typing as npt

"""

    current_docstring = []
    missing_docstrings = [] if check_docstrings else None

    nb_enums = 0
    nb_classes = 0
    nb_functions = 0

    idx_line = 0
    while idx_line < len(pyx_content):

        # This is a docstring
        idx_line, docstring = extract_docstring(pyx_content, idx_line)

        # This is a docstring
        if len(docstring) > 0:
            current_docstring = format_docstring(docstring)

        # This is an enum
        elif match := ENUM_PATTERN.match(pyx_content[idx_line]):
            enum_indent = match.group(1)
            enum_name = match.group(2)
            nb_enums += 1
            idx_line, enum_stub = extract_stub_from_enum(enum_name, len(enum_indent), pyx_content, idx_line, current_docstring, missing_docstrings)
            stub += enum_stub + "\n"

        # This is a class
        elif match := CLASS_PATTERN.match(pyx_content[idx_line]):
            class_indent = match.group(1)
            class_name = match.group(2)
            nb_classes += 1
            idx_line, class_stub = extract_stub_from_class(class_name, len(class_indent), pyx_content, idx_line, current_docstring, missing_docstrings)
            stub += class_stub + "\n"

        # This is a function
        elif (match := FUNCTION_PATTERN.match(pyx_content[idx_line])):
            method_name = match.group(2)
            nb_functions += 1
            idx_line, arg_signature, ret_signature = extract_signature_from_function(pyx_content, idx_line)
            stub += f"def {method_name}({arg_signature}) -> {ret_signature}:\n"
            if len(current_docstring) > 0:
                stub += " " * 4 + "\"\"\"\n"
                for line_docstring in current_docstring:
                    stub += " " * 4 + line_docstring + "\n"
                stub += " " * 4 + "\"\"\"\n"
                current_docstring = []
            elif missing_docstrings is not None and not method_name.startswith("_"):
                missing_docstrings.append(f"Function '{method_name}'")
            stub += f"    {get_stubed_body(ret_signature)}\n\n"

        # Something else (empty line, comment, ...)
        else:
            idx_line += 1

    # Write the stub to file
    with open(output_file_path, 'w', encoding='utf-8') as f:
        f.write(stub)

    print(f"Extracted {nb_enums} enums, {nb_classes} classes, and {nb_functions} functions")
    print(f"Stub file created: {output_file_path}")
    return missing_docstrings if missing_docstrings is not None else []


def format_docstring(docstring: List[str]) -> List[str]:
    """
    Format the docstring to be compliant with Python standards.

    Args:
        docstring: The docstring to format, as a list of lines.

    Returns:
        The formatted docstring, as a list of lines.
    """
    # Doxygen to RST conversions with simple replacements
    replacements = [
        (re.compile(r"\\b\s+"), "**"),
        (re.compile(r"\[([\w\.]+)\]\([\w\s\\]+\)"), r"\1"),
        (re.compile(r"\\ref\s+[\w\.]+\s+\"([\w\.]+)\""), r"\1"),
        (re.compile(r"\\ref\s"), ""),
        (re.compile(r"\\attention\s?"), ".. attention:: "),
        (re.compile(r"\\brief\s?"), ""),
        (re.compile(r"<code>(.*?)</code>"), r"``\1``"),
        (re.compile(r"<b>(.*?)</b>"), r"**\1**"),
        (re.compile(r"<a href=\"(.*?)\">(.*?)</a>"), r"`\2 <\1>`_"),
        (re.compile(r"<ul>"), ""),
        (re.compile(r"</ul>"), ""),
        (re.compile(r"<li>(.*?)</li>"), r"* \1"),
    ]

    # Patterns for warning/warn/note handling
    warning_pattern = re.compile(r"\\warning\s?")
    warn_pattern = re.compile(r"\\warn\s?")
    note_pattern = re.compile(r"\\note\s?")

    # Regex to get the number of spaces the line starts with
    indent_pattern = re.compile(r"^(\s+)")

    # Regex to know if the line start if a \note
    note_pattern_from_beggining_line = re.compile(r"^\s*\\note")

    # Regex to identify a markdown table separator line.
    separator_regex = re.compile(r'^\s*\|\s*---+\s*\|')

    # Regex to identify a table row (line containing table separators)
    table_row_regex = re.compile(r'^\s*\|.*\|\s*$')

    # Process docstring
    new_docstring = []
    is_in_code_block = False
    indent_in_code_block = ""

    for line in docstring:
        # Check if this line is part of a table
        is_table_line = table_row_regex.match(line)

        # Apply general replacements
        for pattern, replacement in replacements:
            line = pattern.sub(replacement, line)

        # Handle some patterns with different treatment for tables vs regular text
        if is_table_line:
            # In tables, replace with simpler text to preserve formatting
            line = warning_pattern.sub("Warning: ", line)
            line = warn_pattern.sub("Warning: ", line)
            line = note_pattern.sub("Note: ", line)
            line = line.replace(r"\n", " ")
        else:
            # In regular text, use RST directive format
            line = warning_pattern.sub(".. warning:: ", line)
            line = warn_pattern.sub(".. warning:: ", line)

        # Number of indent the line starts with
        indent = indent_match.group(0) if (indent_match := indent_pattern.match(line)) else ""

        # Handle code blocks
        if r"\code" in line:
            is_in_code_block = True
            indent_in_code_block = indent
            new_docstring.append(indent + ".. code-block:: text")
            new_docstring.append("")
            continue
        elif r"\endcode" in line:
            is_in_code_block = False
            new_docstring.append("")
            continue

        if is_in_code_block:
            new_docstring.append(indent_in_code_block + "    " + line)
            continue

        # Handle note
        if r"\note" in line:
            if note_pattern_from_beggining_line.match(line):
                *_, line = re.split(r"(\\note)", line, maxsplit=1)
                new_docstring.append(indent + ".. note::")
                new_docstring.append(indent + "    " + line.replace(r"\note", "Note:"))
            else:
                new_docstring.append(indent + line.replace(r"\note", "Note:"))
            new_docstring.append("")
            continue

        # Ignore certain directives
        if any(d in line for d in [r"\ingroup", r"\image"]):
            continue

        # Handle table
        if separator_regex.match(line):
            # Replace any sequence of one or more dashes with ':---:'
            formatted_line = re.sub(r'-+', ':---:', line)
            new_docstring.append(formatted_line)
            continue

        # Handle params and returns
        param_match = re.match(r"\s*\\param\s*(\w+)\s*(?:\[(in|out)\])?\s*:(.*)", line)
        if param_match:
            name, direction, desc = param_match.groups()
            new_docstring.append(f"{indent}:param {name}: {desc.strip()}")
            if direction:
                new_docstring[-1] += f" (Direction: {direction})"
            continue

        return_match = re.match(r"\s*\\return\s*(.*)", line)
        if return_match:
            new_docstring.append(f"{indent}:return: {return_match.group(1).strip()}")
            continue

        new_docstring.append(line)

    if new_docstring and new_docstring[-1] == "":
        new_docstring.pop()

    return new_docstring


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract Python stub definitions from a .pyx file")
    parser.add_argument("input_file", help="Path to the input .pyx file")
    parser.add_argument("output_file", nargs="?", help="Path to the output .pyi file")
    parser.add_argument("--check-docstrings", action="store_true", help="Check for missing docstrings")

    args = parser.parse_args()

    missing = extract_stub_from_pyx(args.input_file, args.output_file, args.check_docstrings)

    if args.check_docstrings and len(missing) > 0:
        print("\nERROR: Missing docstrings for the following:")
        for m in missing:
            print(f" - {m}")
        sys.exit(1)

    sys.exit(0)
