from http import HTTPStatus


def is_http_ok(status_code: int) -> bool:
    return status_code == HTTPStatus.OK


def is_http_success(status_code: int) -> bool:
    return HTTPStatus.OK <= status_code < HTTPStatus.MULTIPLE_CHOICES


def is_retryable_server_error(status_code: int) -> bool:
    return HTTPStatus.INTERNAL_SERVER_ERROR <= status_code < 600
