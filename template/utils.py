import yaml
from redmail import outlook

def send_email(subject, content):
    mail_password = open("./email-password.txt", "r")

    outlook.username = mail_password.readline()[:-1]
    outlook.password = mail_password.readline()[:-1]

    outlook.send(
        receivers=[mail_password.readline()[:-1]],
        subject=subject,
        text=content
    )

    mail_password.close()

def get_config(yaml_config_filename):
    with open(yaml_config_filename) as f:
        config_dict = yaml.safe_load(f)

    return config_dict
