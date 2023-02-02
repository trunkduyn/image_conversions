from dotenv import find_dotenv, load_dotenv

def load_env():
    dotenv_path = find_dotenv(".env.prd")
    if dotenv_path:
        load_dotenv(dotenv_path)

    load_dotenv(find_dotenv(raise_error_if_not_found=True))
