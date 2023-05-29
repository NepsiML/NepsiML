ACTIONS=`git status | grep -e deleted -e modified`

pull:
	git pull
push:
	git add *
	git commit -m "${USER}-${ACTIONS}"
	git push