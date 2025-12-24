import logging
from contextlib import redirect_stdout
import io

def logger_info(func, log_name, **kwargs):
    output_buffer = io.StringIO()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filename=log_name,
        filemode='w',
        encoding='UTF-8'
    )
    with redirect_stdout(output_buffer):
        result = func(**kwargs)  # Запускаем функцию
    output = output_buffer.getvalue()
    logging.info(f"Вывод функции {func.__name__}:\n{output}")
    return result