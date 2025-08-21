import json
import os
import uuid
from typing import Dict, List, Any

class RoundRobinQueueManager:
    def __init__(self):
        self.project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        self.edit_mode_path = os.path.join(self.project_root, 'edit_mode.json')
        self.queue_path = os.path.join(self.project_root, 'user_queue.json')
        
    def load_edit_mode(self) -> Dict:
        """Load edit mode JSON file"""
        try:
            with open(self.edit_mode_path, 'r') as f:
                content = f.read().strip()
                if not content:
                    return {}
                return json.loads(content)
        except FileNotFoundError:
            return {}
        except json.JSONDecodeError as e:
            print(f"Warning: Invalid JSON in edit_mode.json: {e}")
            return {}
    
    def save_edit_mode(self, edit_mode: Dict) -> None:
        """Save edit mode JSON file"""
        with open(self.edit_mode_path, 'w') as f:
            json.dump(edit_mode, f, indent=2)
    
    def load_queue(self) -> Dict:
        """Load user queue JSON file"""
        try:
            with open(self.queue_path, 'r') as f:
                content = f.read().strip()
                if not content:
                    return {}
                return json.loads(content)
        except FileNotFoundError:
            return {}
        except json.JSONDecodeError as e:
            print(f"Warning: Invalid JSON in user_queue.json: {e}")
            return {}
    
    def save_queue(self, queue: Dict) -> None:
        """Save user queue JSON file"""
        with open(self.queue_path, 'w') as f:
            json.dump(queue, f, indent=2)
    
    def generate_job_id(self) -> str:
        """Generate unique job ID"""
        return f"job_{uuid.uuid4().hex[:8]}"
    
    def process_user_requests(self, user_requests: Dict[str, List[str]]) -> Dict[str, str]:
        """
        Process multiple requests per user in round-robin fashion
        
        Args:
            user_requests: Dict with user_id as key and list of requests as value
            Example: {"user1": ["frontend dev", "backend dev"], "user2": ["fullstack dev"]}
        
        Returns:
            Dict with user_id as key and first request as value for current processing
        """
        
        # Load existing edit mode
        edit_mode = self.load_edit_mode()
        
        # Load existing queue
        queue = self.load_queue()
        
        # Dict to store current processing requests (first request of each user)
        current_processing = {}
        
        # Process each user's requests
        for user_id, requests in user_requests.items():
            if not requests:
                continue
                
            # Set user as free in edit mode
            if user_id not in edit_mode:
                edit_mode[user_id] = {}
            if "free" not in edit_mode[user_id]:
                edit_mode[user_id]["free"] = True
            elif edit_mode[user_id].get("free")==False:
                continue
            # First request goes to current processing
            current_processing[user_id] = requests[0]
            
            # Remaining requests go to queue
            if len(requests) > 1:
                queue[user_id] = requests[1:]  # All requests except first
            else:
                # No pending requests, remove from queue if exists
                if user_id in queue:
                    del queue[user_id]
        
        # Save updated edit mode and queue
        self.save_edit_mode(edit_mode)
        self.save_queue(queue)
        
        print(f"✅ Edit Mode Updated: {edit_mode}")
        print(f"✅ Current Processing: {current_processing}")
        print(f"✅ Queue Updated: {queue}")
        
        return current_processing
    
    def get_next_request_for_user(self, user_id: str) -> str | None:
        """
        Get next request for a specific user from queue
        
        Args:
            user_id: User ID to get next request for
            
        Returns:
            Next request string or None if no more requests
        """
        queue = self.load_queue()
        
        if user_id in queue and queue[user_id]:
            next_request = queue[user_id].pop(0)  # Get first request from queue
            
            # Update queue file
            if not queue[user_id]:  # If queue is empty after popping
                del queue[user_id]
            
            self.save_queue(queue)
            print(f"✅ Next request for {user_id}: {next_request}")
            print(f"✅ Updated queue: {queue}")
            
            return next_request
        
        return None
    
    def mark_user_busy(self, user_id: str) -> None:
        """Mark user as busy in edit mode"""
        edit_mode = self.load_edit_mode()
        if user_id in edit_mode:
            edit_mode[user_id]["free"] = False
            self.save_edit_mode(edit_mode)
            print(f"✅ User {user_id} marked as busy")
    
    def mark_user_free(self, user_id: str) -> None:
        """Mark user as free in edit mode"""
        edit_mode = self.load_edit_mode()
        if user_id in edit_mode:
            edit_mode[user_id]["free"] = True
            self.save_edit_mode(edit_mode)
            print(f"✅ User {user_id} marked as free")
    
    def get_user_queue_status(self, user_id: str) -> Dict:
        """Get current status of user's queue"""
        queue = self.load_queue()
        edit_mode = self.load_edit_mode()
        
        return {
            "user_id": user_id,
            "pending_requests": queue.get(user_id, []),
            "pending_count": len(queue.get(user_id, [])),
            "is_free": edit_mode.get(user_id, {}).get("free", True)
        }
    
    def get_all_queue_status(self) -> Dict:
        """Get status of all users' queues"""
        queue = self.load_queue()
        edit_mode = self.load_edit_mode()
        
        status = {}
        all_users = set(list(queue.keys()) + list(edit_mode.keys()))
        
        for user_id in all_users:
            status[user_id] = {
                "pending_requests": queue.get(user_id, []),
                "pending_count": len(queue.get(user_id, [])),
                "is_free": edit_mode.get(user_id, {}).get("free", True)
            }
        
        return status
    
    def clear_user_queue(self, user_id: str) -> None:
        """Clear all pending requests for a user"""
        queue = self.load_queue()
        if user_id in queue:
            del queue[user_id]
            self.save_queue(queue)
            print(f"✅ Cleared queue for user {user_id}")
    
    def clear_all_queues(self) -> None:
        """Clear all user queues"""
        self.save_queue({})
        print("✅ All queues cleared")


# Example usage and testing
def test_round_robin_logic():
    """Test the round-robin queue management"""
    manager = RoundRobinQueueManager()
    
    # Test data
    user_requests = {
        "user1": ["frontend developer", "backend engineer", "data scientist"],
        "user2": ["full stack developer", "mobile developer"],
        "user3": ["DevOps engineer"]
    }
    
    print("🚀 Testing Round-Robin Queue Management")
    print("=" * 50)
    
    # Process initial requests
    current_processing = manager.process_user_requests(user_requests)
    print(f"📋 Current Processing: {current_processing}")
    
    # Get status
    print("\n📊 Queue Status:")
    status = manager.get_all_queue_status()
    for user_id, user_status in status.items():
        print(f"   {user_id}: {user_status}")
    
    # Simulate processing completion and getting next requests
    print("\n🔄 Processing Next Requests:")
    
    # Get next request for user1
    next_req = manager.get_next_request_for_user("user1")
    print(f"   User1 next: {next_req}")
    
    # Get next request for user2
    next_req = manager.get_next_request_for_user("user2")
    print(f"   User2 next: {next_req}")
    
    # Get next request for user3 (should be None)
    next_req = manager.get_next_request_for_user("user3")
    print(f"   User3 next: {next_req}")
    
    # Final status
    print("\n📊 Final Queue Status:")
    status = manager.get_all_queue_status()
    for user_id, user_status in status.items():
        print(f"   {user_id}: {user_status}")


if __name__ == "__main__":
    test_round_robin_logic()
