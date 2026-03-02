from rich.console import Console
from rich.theme import Theme

# Set up custom theme for each agent
custom_theme = Theme({
    "orchestrator": "cyan bold",
    "web_agent": "blue bold",
    "doc_agent": "green bold",
    "fact_checker": "yellow bold",
    "writer": "magenta bold",
    "system": "white dim",
    "error": "red bold"
})

console = Console(theme=custom_theme)

class AgentLogger:
    @staticmethod
    def orchestrator(message: str):
        console.print(f"[orchestrator]🧠 [Orchestrator][/orchestrator] {message}")

    @staticmethod
    def web(message: str):
        console.print(f"[web_agent]🌐 [Web Agent][/web_agent] {message}")

    @staticmethod
    def doc(message: str):
        console.print(f"[doc_agent]📄 [Doc Agent][/doc_agent] {message}")

    @staticmethod
    def fact(message: str):
        console.print(f"[fact_checker]🔍 [Fact Checker][/fact_checker] {message}")

    @staticmethod
    def writer(message: str):
        console.print(f"[writer]✍️  [Writer Agent][/writer] {message}")

    @staticmethod
    def system(message: str):
        console.print(f"[system]⚙️  [System][/system] {message}")

    @staticmethod
    def error(message: str):
        console.print(f"[error]❌ [Error][/error] {message}")
        
    @staticmethod
    def success(message: str):
        console.print(f"[green]✅ {message}[/green]")
