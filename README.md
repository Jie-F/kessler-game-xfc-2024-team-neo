# Neo - Kessler Controller

## About

Neo is a controller I made for the XFC 2024 and WCCI 2024 competitions. It's inspired by Neo from The Matrix. It uses a randomized simulation to perform random search (and a baby version of a genetic algorithm) to perform maneuvers and plan good actions. It is also capable of dodging fast-moving asteroids.

The explainable fuzzy competition's main website is at: [https://xfuzzycomp.github.io/XFC/](https://xfuzzycomp.github.io/XFC/)

To see Neo compete in XFC, see my presentation and matches from XFC 2024 [here](https://www.youtube.com/watch?v=27L8-dkrL-I&t=1627s)

And see Neo's WCCI competition [here](https://www.youtube.com/watch?v=863WyDLXvNI). Neo was made significantly better, by fixing a bug, and also by making it attack the other ship, taking away its lives and taking all the asteroids for itself. It also purposely runs into asteroids at the end of the game, to exchange a life for an extra point.

Feel free to look through my code for ideas, and feel free to use any of this for yourself! I'm no longer competing in XFC since unfortunately I graduated :(

## Repo

This repo was forked from the [main kesslergame repo](https://github.com/ThalesGroup/kessler-game/tree/main/src/kesslergame)

Neo's code is all in neo_controller.py. Competition specific versions were created and put in the SUBMISSIONS folders for XFC 2024 and WCCI 2024. Those versions of neo_controller.py were derived by running the competition preprocessor to strip out unnecessary code. And then the controller was compiled with MyPyC into a compiled Python module to get a 5-10X speedup in performance. You can run the controller in interpreted Python mode, but it will be extremely slow, and Neo also self-regulates its search iterations to control the lag, meaning the ingame performance will also suffer.

Additionally, I also have scenarios you can use to test on:
- custom_scenarios.py have a ton of crazy scenarios I made
- adversarial_scenarios_for_jie.py are xfc 2024 scenarios from Tim
- scenarios.py are official xfc 2021 scenarios from their repo
- xfc_2023_replica_scenarios.py were made by me manually

## Contact

Feel free to contact me at jie.f@pm.me with any questions!
