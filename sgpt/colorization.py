import re

from pygments import highlight
from pygments.formatters.terminal import TerminalFormatter
from pygments.lexers.python import PythonLexer


def colorize_code(text):
    pattern = r'```python\n(.*?)\n```'
    code_blocks = re.findall(pattern, text, re.DOTALL)
    for code in code_blocks:
        colored_code = highlight(code, PythonLexer(), TerminalFormatter())
        colored_code = "<=======================================\n" + colored_code + "=======================================>\n"
        text = text.replace(f'```python\n{code}\n```', colored_code)

    return text
