document.addEventListener("DOMContentLoaded", () => {
    
    // Drag and Drop Elements
    const dropZone = document.getElementById("drop-zone");
    const fileInput = document.getElementById("file-input");
    const filePreview = document.getElementById("file-preview");
    const fileNameSpan = document.getElementById("file-name");
    const imgThumbnail = document.getElementById("img-thumbnail");
    const uploadForm = document.getElementById("uploadForm");
    const loadingState = document.getElementById("loading");
    
    // Only execute if we are on a page with the drag and drop functionality
    if (dropZone && fileInput) {
        
        // Click to open file dialog
        dropZone.addEventListener("click", () => {
            fileInput.click();
        });

        // Drag events
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            dropZone.addEventListener(eventName, preventDefaults, false);
        });

        function preventDefaults (e) {
            e.preventDefault();
            e.stopPropagation();
        }

        // Highlight effect
        ['dragenter', 'dragover'].forEach(eventName => {
            dropZone.addEventListener(eventName, () => {
                dropZone.classList.add("dragover");
            }, false);
        });

        ['dragleave', 'drop'].forEach(eventName => {
            dropZone.addEventListener(eventName, () => {
                dropZone.classList.remove("dragover");
            }, false);
        });

        // Handle file drop
        dropZone.addEventListener('drop', (e) => {
            const dt = e.dataTransfer;
            const files = dt.files;
            
            if (files.length > 0) {
                fileInput.files = files; // Assign files to input
                handleFiles(files[0]);
            }
        });

        // Handle file select from dialog
        fileInput.addEventListener('change', (e) => {
            if (fileInput.files.length > 0) {
                handleFiles(fileInput.files[0]);
            }
        });

        // Show preview logic
        function handleFiles(file) {
            // Check if it's an image
            if (!file.type.match('image.*')) {
                alert("Please upload an image file (JPEG, PNG).");
                return;
            }

            // Update UI with file details
            fileNameSpan.textContent = file.name;
            dropZone.classList.add("d-none"); // Hide dropzone
            filePreview.classList.remove("d-none"); // Show preview wrapper
            
            // Read file for thumbnail
            const reader = new FileReader();
            reader.onload = (e) => {
                imgThumbnail.src = e.target.result;
                imgThumbnail.classList.remove("d-none");
                imgThumbnail.classList.add("fade-in");
            }
            reader.readAsDataURL(file);
        }

        // Handle form submission and loading spinner
        uploadForm.addEventListener('submit', (e) => {
            if (fileInput.files.length === 0) {
                e.preventDefault();
                alert("Please select an image to analyze.");
                return;
            }
            
            // Hide the entire form
            uploadForm.classList.add("d-none");
            // Show loading state
            loadingState.classList.remove("d-none");
            loadingState.classList.add("fade-in");
            
            // Allow form to submit normally so the server handles redirect
        });
    }
});
