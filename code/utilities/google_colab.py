def is_running_on_colab():
    try:
        import google.colab
        return True
    except ImportError:
        return False