# -*- coding: utf-8 -*-
"""Helper toolbox for adaptive sampling."""


__LOGGING_ENABLED__ = True # Set to False to disable logging and use print instead.
DEFAULT_LOGGER_NAME = "adasamp"
if __LOGGING_ENABLED__:
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(DEFAULT_LOGGER_NAME)
    

def log_wrapper(verbose, level, msg):
    """Wrapper to safely log messages.
     
    Use the ``logging`` module if enabled and use ``print`` to std otherwise.
    
    Parameters
    ----------
    
    verbose : bool
        Set to True to log messages.
        
    level : int
        Logging level for the ``logging`` module.
    
    msg : str
        Actual message to log.
    """

    if verbose:
        try:
            if __LOGGING_ENABLED__:      
                logging.log(int(level), str(msg))
            else:
                print(str(msg))
        except Exception as e:
            print(e)