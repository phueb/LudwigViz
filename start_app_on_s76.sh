#!/usr/bin/env bash

SERVER_NAME='s76'

# rsync python files to server
echo RSyncing python files to ${SERVER_NAME}
rsync --verbose --recursive --stats ludwigviz ${SERVER_NAME}:/home/ph/ludwigviz

ssh -X ${SERVER_NAME} <<- EOF
    APP_PID=\$(lsof -i:5000 -t)
    if [ \${APP_PID:-0} -gt 0 ]
    then
        echo Killing LudwigViz on port 5000.
        kill -9  \${APP_PID}
    fi
    cd /home/ph/ludwigviz/
    python3 app.py  --nodebug
EOF
