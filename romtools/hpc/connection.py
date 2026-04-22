import os
import subprocess
import tempfile

class Result:
    """
    Simple class to encapsulate the result of a command execution.
    """
    def __init__(self, stdout, stderr, exited):
        self.stdout = stdout
        self.stderr = stderr
        self.exited = exited
        self.ok = exited == 0

class Connection:
    """
    A simple SSH connection class that uses subprocess to execute SSH and SCP commands.

    This uses connection multiplexing (ControlMaster) to reuse an existing connection
    per SSH session (thus only requiring authentication & handshakes once).
    """
    def __init__(self, host, user=None, port=None, persist_seconds=600):
        self.host = host
        self.user = user
        self.port = port if port else 22

        self.target = f"{self.user}@{self.host}" if self.user else self.host

        # Use a temp directory to hold the control socket.
        # ControlPath has length limits on some systems, so keep it short.
        sock_dir = tempfile.mkdtemp(prefix="sshcm-")
        self.control_path = os.path.join(sock_dir, "cm.sock")
        self.common = [
                "ssh",
                "-o", "StrictHostKeyChecking=accept-new",
                "-o", f"ControlPath={self.control_path}",
            ]

        # Start master connection, prompts for password once
        subprocess.run(self.common + [
                "-o", "ControlMaster=yes",
                "-o", f"ControlPersist={persist_seconds}",
                "-p", str(self.port),
                self.target,
                "-N"  # no remote command; just hold connection
            ])

    def run(self, command):
        """Execute a command on the remote host"""
        ssh_cmd = [
            "ssh",
            "-o", f"ControlPath={self.control_path}",
            "-p", str(self.port),
            self.target,
            command
        ]

        result = subprocess.run(
            ssh_cmd,
            capture_output=True,
            text=True
        )

        return Result(result.stdout, result.stderr, result.returncode)

    def get(self, remote, local=None):
        """Download a file from the remote host"""
        remote_file = f"{self.target}:{remote}"
        local_path = local if local else remote.split("/")[-1]
        local_dir = os.path.dirname(local_path)
        if local_dir:
            os.makedirs(local_dir, exist_ok=True)
        scp_cmd = [
            "scp",
            "-o", f"ControlPath={self.control_path}",
            "-P", str(self.port),
            remote_file,
            local_path
        ]

        result = subprocess.run(scp_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"SCP failed: {result.stderr}")

    def put(self, local, remote=None):
        """Upload a file to the remote host"""
        remote_path = remote if remote else os.path.basename(local)
        remote_file = f"{self.target}:{remote_path}"
        scp_cmd = [
            "scp",
            "-o", f"ControlPath={self.control_path}",
            "-P", str(self.port),
            local,
            remote_file
        ]

        result = subprocess.run(scp_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"SCP failed: {result.stderr}")

    def local(self, cmd):
        """Execute a command locally"""
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True
        )
        return Result(result.stdout, result.stderr, result.returncode)

    def close(self):
        """Close the SSH connection"""
        subprocess.run([
            "ssh",
            "-o", f"ControlPath={self.control_path}",
            "-p", str(self.port),
            "-O", "exit",
            self.target,
        ])
