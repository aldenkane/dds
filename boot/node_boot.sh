#!/usr/bin/env bash

cd /home/pi/dds/sourceCode
/bin/sleep 30
/usr/local/bin/node parsePi.js # /usr/local/bin/node used for optoa0002. a0001 uses /usr/bin/node. Run which node to find path to command
