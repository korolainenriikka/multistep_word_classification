# Multistep workflow: word classification

notes on current status:
 * current Dockerfile installs docker inside container
 * the volume in MLProject binds word_classifier containers' docker socket to the outer machines' docker socket. The containers created inside the container will be created in the outer machine
 * current error in run command: run not found.
