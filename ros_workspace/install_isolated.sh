#!/usr/bin/env bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
OLD="$PWD"
cd "$DIR"
# Sub-shell makes ^C work properly when sourced
(catkin_make_isolated --install --use-ninja)
if [[ "${BASH_SOURCE[0]}" != "${0}" ]]; then
	echo "script ${BASH_SOURCE[0]} is being sourced ..."
	echo "Will source install_isolated/setup.bash"
	source install_isolated/setup.bash
else
	echo "Remember to source install_isolated/setup.(bash|zsh|sh)"
fi
cd "$PWD"
