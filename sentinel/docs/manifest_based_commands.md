# Manifest-based Commands: Developer Guide

Overview
- Manifest-based commands allow adding new capabilities to SentinelAI without modifying core code.
- Each command lives as a small plugin directory containing a manifest.json and a plugin.py module.

Directory structure
- sentinel/plugins/
  - YourPluginName/
    - manifest.json
    - plugin.py

Manifest schema (manifest.json)
- name: Friendly display name of the plugin
- version: Semantic version (e.g., 1.0.0)
- description: Short description of what the plugin does
- author: Author/maintainer
- intents: ["intent_name"] (informational; matches intents used by routing, optional)
- triggers: ["phrase 1", "phrase 2"] (user-visible trigger phrases)
- entry: "plugin.py" (module file that contains the class)
- class: "ClassName" (the class defined in plugin.py to instantiate)
- requires: ["dep1>=version"] (optional dependencies)

Plugin class conventions
- The loader expects a class named exactly as manifest.class in the module defined by manifest.entry.
- The class should implement a constructor accepting an optional orchestrator, and an execute(command, entity) method returning a string.
- Best practice: subclass an existing core plugin when you want to reuse patterns and avoid duplicating logic.

Loading behavior (how it works at runtime)
- CommandRouter._load_manifest_plugins discovers sentinel/plugins/*, reads manifest.json, loads plugin.py, and instantiates the class named in manifest.class.
- If a dependency listed under requires cannot be installed, loading may fail gracefully depending on environment.
- There can be multiple manifest plugins loaded alongside core plugins; ensure unique plugin names to avoid collisions in the registry.

Implementation example (existing samples)
- PowerManagement (Sleep/Hibernate) and Brightness are already added as manifest-based plugins in this repo.
- Example snippet for your own plugin:
  - manifest.json
    {
      "name": "Your Plugin",
      "version": "1.0.0",
      "description": "Description of your command",
      "author": "Your Name",
      "intents": ["some_intent"],
      "triggers": ["your trigger phrase"],
      "entry": "plugin.py",
      "class": "YourPluginClass",
      "requires": []
    }
  - plugin.py
    from sentinel.commands.some_base import SomeBase  # if you prefer to subclass
    class YourPluginClass(SomeBase):
        def __init__(self, orchestrator=None):
            super().__init__(orchestrator)
        def execute(self, command: str, entity: str) -> str:
            return "Your plugin response"

Testing and debugging tips
- Run your app and monitor logs for messages like: "Manifest plugin system: N extension(s) loaded."
- Use unit tests to load the manifest and verify that the plugin class is instantiated and responds to a sample command.
- Ensure your plugin’s trigger phrases are distinct to avoid conflicts with core plugins.

Governance and compatibility
- Follow semantic versioning for manifest plugins.
- If removing a plugin in a future release, provide a deprecation path and clear user messaging.
- Maintain Windows-centric behavior unless you explicitly introduce cross-platform helpers.
