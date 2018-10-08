#!/usr/bin/env bash

SERVER_NAME='s76'

# rsync python files to server
echo RSyncing python files to ${SERVER_NAME}
rsync --verbose --recursive --stats ludwiglab ${SERVER_NAME}:/home/lab/ludwiglab

ssh -X ${SERVER_NAME} <<- EOF
    APP_PID=\$(lsof -i:5000 -t)
    if [ \${APP_PID:-0} -gt 0 ]
    then
        echo Killing LudwigLab on port 5000.
        kill -9  \${APP_PID}
    fi
    cd /home/lab/ludwiglab/
    python3 flask_app.py  --nodebug
EOF
