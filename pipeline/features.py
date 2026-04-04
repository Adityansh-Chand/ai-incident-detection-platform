def extract_features(log):

    return [
        len(log),
        log.count("error"),
        log.count("timeout")
    ]
