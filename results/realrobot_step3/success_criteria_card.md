# Real-Robot Success And Retry Criteria

Success: the object leaves the table by at least 10 cm and remains held for at least 5 s.

Retry rule: only hardware fault trials may be retried. Grasp failures are not retried. Every retry must keep the original trial ID and add a retry note.

Failure codes:
A = no contact / empty grasp
B = slips after contact before lift
C = drops during lift
D = drops during 5 s hold
E = collision or safety stop
F = hardware fault retry

Initial placement: each object uses a fixed marked pose. Photograph the placement before the first trial of each object block and whenever reposed.
