from telco_agent.api.dependencies import get_ingestion_service, get_settings


def main() -> None:
    settings = get_settings()
    ingested = get_ingestion_service().ingest(settings.knowledge_base_dir)
    print(f"Ingested {ingested} knowledge chunks into '{settings.qdrant_collection}'.")


if __name__ == "__main__":
    main()
