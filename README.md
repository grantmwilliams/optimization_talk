# optimization_talk
The code in this repo was written for the [Comparison of Optimization Methods](https://www.meetup.com/Tulsa-Data-Science-Meetup/events/240312307/) talk for the Tulsa Data Science meetup.

The Optimization Library itself was written with simplicity in mind and focuses on readability for beginners to these algorithms over efficiency. In the future I would like to update the library to include efficient solutions of some of the different algorithms.

The Library does not handle out of bounds on the algorithms, but an easy fix to this problem would be to simply assign a very poor fitness value to any out of bounds coordinate and let the function move away from the bad value.

