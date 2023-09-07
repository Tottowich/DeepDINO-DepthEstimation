# 📘 Totto-style Docstring Guidelines

Hello, dear developer! 🌟 If you're looking to contribute to the Totto project, please follow these docstring guidelines. They're based on the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings) and the [NumPy Docstring Guide](https://numpydoc.readthedocs.io/en/latest/format.html#docstring-standard).

---

## 🌐 Table of Contents

- [📘 Totto-style Docstring Guidelines](#-totto-style-docstring-guidelines)
  - [🌐 Table of Contents](#-table-of-contents)
  - [📁 Module-Level Docstrings](#-module-level-docstrings)
  - [🏛 Class-Level Docstrings](#-class-level-docstrings)
  - [🛠 Method \& Function-Level Docstrings](#-method--function-level-docstrings)
    - [Function Example:](#function-example)
    - [Method Example:](#method-example)
  - [🧠 Data Analysis \& ML Specifics](#-data-analysis--ml-specifics)
  - [Docformating Prompt](#docformating-prompt)

---

## 📁 Module-Level Docstrings

Every module should tell its own story. What's its purpose? What classes and functions does it house?

``` python
"""
module_name.py
==============

This module provides utility functions and classes for a specific task.

Classes:
--------
ExampleClass : A brief description of the class.

Functions:
----------
example_function : A brief description of the function.
"""
```

---

## 🏛 Class-Level Docstrings

Classes can be complex. Describe its purpose, its ancestry, and its methods.

``` python
class ExampleClass(ParentClass):
    """
    This class is designed to perform a specific task.
    
    Inherits from:
    --------------
    ParentClass from (module_name.py)
    
    Methods:
    --------
    example_method : A brief description of the methods.
    
    Attributes:
    -----------
    example_attribute : A brief description of the attributes (if any).

    Note:
    -----
    Any important note about the class.

    """
```

---

## 🛠 Method & Function-Level Docstrings

For methods and functions, the parameters, return values, and special notes are crucial.

### Function Example:

``` python
def example_function(param1: Type, param2: Type) -> ReturnType:
    """
    A brief description of what the function does.
    
    Parameters:
    -----------
    param1 : Type
        Description of param1.
    param2 : Type
        Description of param2.
    
    Returns:
    --------
    ReturnType
        Description of the return value.
    
    Raises:
    -------
    ExceptionType
        When the exception is raised.
    """
```

### Method Example:

``` python
def example_method(self, param1: Type) -> None:
    """
    A brief description of what the method does.
    
    Parameters:
    -----------
    param1 : Type
        Description of param1.
    
    Note:
    -----
    Any important note about the method.
    """
```

---

## 🧠 Data Analysis & ML Specifics

For data analysis and ML projects, documentation needs a bit more detail:

- **Data Input/Output**: Clearly document the expected input data format and the format of the output.
  
- **Model Parameters**: If your class/method deals with ML models, describe hyperparameters, training parameters, and any other model-specific parameters.

- **Metrics**: Clearly define any metrics that your functions/classes compute.

- **Dependencies**: If certain methods depend on specific libraries (e.g., TensorFlow, PyTorch, scikit-learn), make a note of it.

- **Examples**: Provide small examples, especially for data preprocessing, model training, or evaluation functions. This helps users quickly understand the functionality.

---

Remember, clear and consistent documentation makes collaboration smoother and errors fewer. Let's keep our code clean and our comments cleaner! Happy coding and analyzing! 🚀


## Docformating Prompt
```python
"""
Based on the Totto-style Docstring Guidelines, read the entire code file and convert all the docstrings to Totto-style. Ensure that the converted docstrings adhere to the Totto-style specifications while maintaining the original functionality described in the existing docstrings.
"""
```
