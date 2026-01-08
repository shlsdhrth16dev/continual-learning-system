"""
Structured logging configuration for the continual learning system.
Provides consistent logging across all modules.
"""
import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

from src.config import get_config


def setup_logging(
    name: Optional[str] = None,
    log_file: Optional[Path] = None,
    level: Optional[str] = None
) -> logging.Logger:
    """
    Setup logging with both console and file handlers.
    
    Args:
        name: Logger name (defaults to root logger)
        log_file: Path to log file (defaults to config)
        level: Logging level (defaults to config)
    
    Returns:
        Configured logger instance
    """
    config = get_config()
    
    # Get or create logger
    logger = logging.getLogger(name)
    
    # Set level
    log_level = level or config.logging.level
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    # Create formatters
    formatter = logging.Formatter(
        config.logging.format,
        datefmt=config.logging.date_format
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if config.logging.enable_file_logging:
        if log_file is None:
            log_file = config.paths.logs_dir / config.logging.log_file
        
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=config.logging.max_bytes,
            backupCount=config.logging.backup_count
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Prevent propagation to root logger
    logger.propagate = False
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger for a specific module.
    
    Args:
        name: Module name (usually __name__)
    
    Returns:
        Logger instance
    """
    return setup_logging(name)
