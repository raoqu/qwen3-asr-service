"""兼容层错误信封：各上游风格独立，按异常类型分派，不污染 v2 的 {"detail":...}。

v2 的 HTTPException 仍由 FastAPI 默认 handler 处理；这里只接管 OpenAICompatError，
按异常**类型**分派（非按路径），对 /v1、/v2 零回归。DashScope 错误信封（Phase 2）
后续并入本模块。
"""
from fastapi import HTTPException
from fastapi.responses import JSONResponse


class OpenAICompatError(HTTPException):
    """OpenAI 风格错误：响应体 {"error":{message,type,param,code}}。"""

    def __init__(self, status_code: int, message: str, *,
                 err_type: str = "invalid_request_error",
                 param: str | None = None, code: str | None = None):
        super().__init__(status_code=status_code, detail=message)
        self.err_type = err_type
        self.param = param
        self.code = code


def register_compat_exception_handlers(app) -> None:
    """注册兼容层异常 handler（按类型分派，不触碰 v2 默认 handler）。"""

    @app.exception_handler(OpenAICompatError)
    async def _handle_openai(_request, exc: OpenAICompatError):
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": {
                "message": exc.detail,
                "type": exc.err_type,
                "param": exc.param,
                "code": exc.code,
            }},
        )
