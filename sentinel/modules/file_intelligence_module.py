import os
import pypdf
import docx
import logging
from datetime import datetime, timedelta

class FileIntelligenceModule:
    def __init__(self, knowledge_base):
        self.knowledge_base = knowledge_base
        self.logger = logging.getLogger("FileIntelligenceModule")

    def scan_directory(self, directory, hours=24):
        """Scans a directory for new documents (PDF, DOCX) modified within the last N hours."""
        now = datetime.now()
        threshold = now - timedelta(hours=hours)
        found_files = []
        try:
            for root, dirs, files in os.walk(directory):
                for file in files:
                    if file.lower().endswith(('.pdf', '.docx')):
                        filepath = os.path.join(root, file)
                        mtime = datetime.fromtimestamp(os.path.getmtime(filepath))
                        if mtime > threshold:
                            found_files.append(filepath)
            
            self.logger.info(f"Found {len(found_files)} new documents in {directory}.")
            return found_files
        except Exception as e:
            self.logger.error(f"Error scanning directory: {e}")
            return []

    def index_document(self, filepath):
        """Reads a document and adds its content to the knowledge base."""
        try:
            content = ""
            if filepath.lower().endswith('.pdf'):
                with open(filepath, 'rb') as f:
                    reader = pypdf.PdfReader(f)
                    for page in reader.pages:
                        content += page.extract_text() or ""
            elif filepath.lower().endswith('.docx'):
                doc = docx.Document(filepath)
                for para in doc.paragraphs:
                    content += para.text + "\n"
            
            if content:
                # Add content to knowledge base for RAG
                filename = os.path.basename(filepath)
                metadata = {"source": "file_intelligence", "filename": filename, "path": filepath}
                # Chunking large documents for vector search
                chunks = [content[i:i+1000] for i in range(0, len(content), 1000)]
                for i, chunk in enumerate(chunks):
                    self.knowledge_base.add_information(
                        f"From file {filename} (chunk {i+1}): {chunk}",
                        metadata=metadata
                    )
                return True, f"Indexed {filename} ({len(chunks)} chunks)."
            return False, f"Could not extract content from {filepath}."
        except Exception as e:
            self.logger.error(f"Error indexing {filepath}: {e}")
            return False, f"Error: {e}"

    def summarize_recent_files(self, directory, hours=24, llm_callback=None):
        """Summarizes all new documents in a directory."""
        files = self.scan_directory(directory, hours=hours)
        if not files:
            return "No new documents found."
        
        summary = "Summary of recent documents:\n"
        for f in files:
            success, msg = self.index_document(f)
            if success and llm_callback:
                # Get a quick summary from the LLM based on the indexed content
                context = self.knowledge_base.query_knowledge(f"What is in the file {os.path.basename(f)}?")
                prompt = f"Summarize the following information from the file {os.path.basename(f)} in one sentence:\n{context}"
                doc_summary = llm_callback(prompt)
                summary += f"- {os.path.basename(f)}: {doc_summary}\n"
            else:
                summary += f"- {os.path.basename(f)}: {msg}\n"
        return summary
