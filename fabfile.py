# -*- encoding: utf-8 -*-

import os

from fabric.api import *

DIGEST_FOLDER = '/home/pythondigest/pythondigest.ru/'
DIGEST_REPO_FOLDER = os.path.join(DIGEST_FOLDER, 'repo')
DIGEST_ENV_FOLDER = os.path.join(DIGEST_FOLDER, 'env')
DIGEST_ENV_PATH = os.path.join(DIGEST_ENV_FOLDER, 'bin/activate')

CLS_FOLDER = '/home/pythondigest/cls.pythondigest.ru/'
CLS_REPO_FOLDER = os.path.join(CLS_FOLDER, 'repo')
CLS_ENV_FOLDER = os.path.join(CLS_FOLDER, 'env')
CLS_ENV_PATH = os.path.join(CLS_ENV_FOLDER, 'bin/activate')

LOCAL_DOWNLOADS_FOLDER = '/home/warmonger/Downloads'
LOCAL_CLS_ENV_PATH = '/home/warmonger/Develop/venv/project_pydigest_an/bin/activate'
LOCAL_CLS_FOLDER = '/home/warmonger/Develop/Projects/pydigest-classifier'
LOCAL_CLS_PYTHON = '/home/warmonger/Develop/venv/project_pydigest_an/bin/python'


def create_dataset():
    with cd(DIGEST_FOLDER), prefix('source %s' % DIGEST_ENV_PATH):
        with cd('./dataset'):
            run('rm -f ./*')

        with cd('./repo'):
            run('python manage.py create_dataset 30 80')

        run('zip -r dataset.zip dataset/')
        run('mv dataset.zip ./media/')


def update_cls():
    env.shell = '/bin/bash'
    with lcd(LOCAL_DOWNLOADS_FOLDER):
        local('rm -f dataset.zip')
        local('rm -rf ./dataset')
        local('wget https://pythondigest.ru/media/dataset.zip')
        local('unzip dataset.zip')

        with lcd(LOCAL_CLS_FOLDER):
            local('%s train.py /home/warmonger/Downloads/dataset/ ./api/classifier_64.pkl' % LOCAL_CLS_PYTHON)
            local('git add api/classifier_64.pkl')
            local('git commit -m "Update cls"')
            local('git push')


def restart_cls():
    with cd(CLS_REPO_FOLDER):
        run('git pull')
        sudo('service uwsgi restart')


def create_report():
    with cd(DIGEST_FOLDER), prefix('source %s' % DIGEST_ENV_PATH):
        run('rm -f ./media/report.csv')
        with cd(DIGEST_REPO_FOLDER):
            run('python manage.py mark_all_cls_off')
            run('python manage.py update_cls_check ../dataset/test_set_ids.txt')
            run("python manage.py create_cls_report '../media/report.csv' ../dataset/test_set_ids.txt")


def analyze_report():
    with cd(LOCAL_DOWNLOADS_FOLDER):
        report_path = '/home/warmonger/Downloads/report.csv'
        local("rm -f %s" % report_path)
        local("wget https://pythondigest.ru/media/report.csv -O %s" % report_path)
        with cd(LOCAL_CLS_FOLDER):
            local("%s report.py %s" % (LOCAL_CLS_PYTHON, report_path))


def update_all_cls():
    with cd(DIGEST_FOLDER), prefix('source %s' % DIGEST_ENV_PATH):
        with cd(DIGEST_REPO_FOLDER):
            run('python manage.py check_all_cls')


def update():
    create_dataset()
    update_cls()


def report():
    # restart_cls()
    # create_report()
    analyze_report()
    # update_all_cls()
