from romtools.hpc.util.logger import Logger


def test_log_without_hostname_prints_plain_message(capsys):
    Logger().log("hello")

    assert capsys.readouterr().out == "hello\n"


def test_log_with_hostname_prefixes_message(capsys):
    logger = Logger()
    logger.set_hostname("cluster1")

    logger.log("hello")

    assert capsys.readouterr().out == "[cluster1] hello\n"


def test_log_local_true_skips_hostname_prefix(capsys):
    logger = Logger()
    logger.set_hostname("cluster1")

    logger.log("hello", local=True)

    assert capsys.readouterr().out == "hello\n"


def test_debug_disabled_prints_nothing(capsys):
    Logger(debug=False).debug("hidden")

    assert capsys.readouterr().out == ""


def test_debug_enabled_prints_with_prefix(capsys):
    Logger(debug=True).debug("shown")

    assert capsys.readouterr().out == "[DEBUG] shown\n"
