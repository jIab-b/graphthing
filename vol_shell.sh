#!/bin/bash
python -c 'from mlrunner import MLRunner; MLRunner(backend="modal", config_path="lang_config.txt").shell(interactive=True, gpu=False)'


