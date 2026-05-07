import json
import os
import urllib.parse
from http.server import SimpleHTTPRequestHandler, HTTPServer
import threading
from dotenv import load_dotenv

# Force load variables from .env to override any stale shell exports
load_dotenv(override=True)

# Import the existing RAG logic
from src.retrieval_engine import create_relationship_aware_rag_chain, get_vectorstore
from src.evaluate import run_evaluation_suite
from src.results_manager import load_eval_results
from src.auth_db import register_user, verify_user, init_db
from src.ingestion import load_and_chunk_pdfs
from src.metadata_tagger import enrich_metadata

# Global state for UI progress tracking
eval_progress = {"status": "idle", "percent": 0, "message": ""}

class RAGDashboardHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        # Allow CORS for development
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    def do_GET(self):
        if self.path == '/':
            self.path = '/frontend/index.html'
        elif self.path.startswith('/api/results'):
            results = load_eval_results()
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(results or {}).encode())
            return
        elif self.path.startswith('/api/eval_progress'):
            global eval_progress
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(eval_progress).encode())
            return
        elif self.path.startswith('/api/graph'):
            graph_path = "data/relationship_graph.json"
            if os.path.exists(graph_path):
                with open(graph_path, "r") as f:
                    self.send_response(200)
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()
                    self.wfile.write(f.read().encode())
            else:
                self.send_response(404)
                self.end_headers()
            return
            
        return super().do_GET()

    def do_POST(self):
        content_length = int(self.headers.get('Content-Length', 0))
        post_data = self.rfile.read(content_length) if content_length > 0 else b""
        
        try:
            payload = json.loads(post_data.decode('utf-8')) if post_data else {}
        except json.JSONDecodeError:
            payload = {}

        import traceback

        if self.path == '/api/chat':
            query = payload.get("query", "")
            strict_mode = payload.get("strict_mode", False)
            try:
                rag_chain = create_relationship_aware_rag_chain(strict_mode=strict_mode)
                response = rag_chain.invoke({"input": query})
                
                answer = response.get("answer", "")
                sources = response.get("context", [])
                
                docs = []
                for d in sources:
                    docs.append({
                        "name": d.metadata.get("document_name", "Unknown"),
                        "status": d.metadata.get("status", "Unknown")
                    })
                
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({
                    "answer": answer,
                    "sources": docs
                }).encode())
            except Exception as e:
                with open("server_error.log", "a") as f:
                    f.write(f"API Chat Error: {str(e)}\n{traceback.format_exc()}\n")
                self.send_response(500)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode())
                
        elif self.path == '/api/evaluate':
            global eval_progress
            if eval_progress.get("status") == "running":
                self.send_response(400)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"error": "Already running"}).encode())
                return
                
            eval_progress = {"status": "running", "percent": 0, "message": "Initializing LLM-as-a-Judge suite..."}
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            
            # Start evaluation in a background thread to prevent blocking
            def progress_callback(p, m):
                global eval_progress
                eval_progress["percent"] = int(p * 100)
                eval_progress["message"] = m

            def run_bg():
                global eval_progress
                try:
                    run_evaluation_suite(progress_callback)
                    eval_progress["status"] = "completed"
                    eval_progress["percent"] = 100
                    eval_progress["message"] = "Evaluation complete! Real metrics generated."
                except Exception as e:
                    eval_progress["status"] = "error"
                    eval_progress["message"] = f"Error: {str(e)}"
            
            threading.Thread(target=run_bg, daemon=True).start()
            self.wfile.write(json.dumps({"status": "Evaluation started in background."}).encode())
            
        elif self.path == '/api/login':
            username = payload.get("username", "")
            password = payload.get("password", "")
            success, role = verify_user(username, password)
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            if success:
                self.wfile.write(json.dumps({"success": True, "role": role, "username": username}).encode())
            else:
                self.wfile.write(json.dumps({"success": False, "error": "Access Denied."}).encode())

        elif self.path == '/api/register':
            username = payload.get("username", "")
            password = payload.get("password", "")
            admin_key = payload.get("admin_key", "")
            is_admin = (admin_key == "ADMIN_123")
            success, msg = register_user(username, password, is_admin)
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"success": success, "message": msg}).encode())
            
        elif self.path == '/api/sync':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            
            def run_sync_bg():
                try:
                    chunks = load_and_chunk_pdfs()
                    tagged = enrich_metadata(chunks)
                    get_vectorstore(tagged)
                except Exception as e:
                    print(f"Sync error: {e}")
            
            threading.Thread(target=run_sync_bg).start()
            self.wfile.write(json.dumps({"success": True, "message": "System Sync initiated in background."}).encode())
            
        else:
            self.send_response(404)
            self.end_headers()

def run(port=8080):
    print("Starting API Server on port 8080...")
    
    # Eagerly initialize the RAG engine so the first request isn't slow
    print("Pre-loading LLM PyTorch models and BM25 index. This may take 5-10 seconds...")
    try:
        create_relationship_aware_rag_chain()
        print("✅ AI Engine pre-loaded successfully! Ready for instant chats.")
    except Exception as e:
        print(f"⚠️ Engine pre-load skipped: {e}")
        
    httpd = HTTPServer(('0.0.0.0', 8080), RAGDashboardHandler)
    print(f"Server is listening on http://localhost:{port}")
    print("Press Ctrl+C to stop.")
    httpd.serve_forever()

if __name__ == '__main__':
    # Ensure frontend dir exists
    os.makedirs("frontend", exist_ok=True)
    init_db()
    run()
