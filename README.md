# Multi-Task Learning

How to run main code:
- Clone the repository
- Create a new environment from `requirements.txt`
- Run `main.py`

How to use this as a library:
- You can import functions from from the different files. The code is designed to behave like a module.
- In particular, the function `get_prebuilt_model` is very useful if you want to load in-built (default) models
- You can add new defaults based on configurations that you build
- It is also possible to build models in a bespoke way


How to run legacy code:
- The legacy code behaves slightly differently to the main code. It is included for reference.
- In particular, `main_colab.py` and `colab_continue_train.py` are useful if you are running on Colab and have trouble with runtimes restarting. You can use them to save models and then re-train from the previously saved state.
- `main.py` from `legacy` should not be used.