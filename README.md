report-analyzer/
├── pyproject.toml
├── setup.py
└── backend/
    ├── __init__.py
    ├── main.py
    ├── api/
    │   ├── __init__.py
    │   └── routes.py
    ├── core/
    │   ├── __init__.py
    │   ├── services/
    │   │   ├── __init__.py
    │   │   ├── analysis_service.py
    │   │   └── cloud_ai_service.py
    │   └── workflows/
    │       ├── __init__.py
    │       └── analysis_graph.py
    ├── config/
    │   ├── __init__.py
    │   ├── settings.py
    │   └── config.yaml
    └── infra/
        ├── __init__.py
        └── cloud/
        |    ├── __init__.py
		|	├── aws_provider.py
		|	├── azure_provider.py
		|	└── gcp_provider.py
		|
		|_____storage/
				|----__init__.py
				|
				|_____vector_store.py

This is expected project structure for backeend , however currently thinking include entire files in backend itseld including project.toml and setup.py. will review and decide on this 				