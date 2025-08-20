(function() {
    function filterTools() {
        const q = document.getElementById('tool-search').value.trim().toLowerCase();
        const cards = document.querySelectorAll('.tool-card');
        let visible = 0;
        cards.forEach(c => {
            if (!q || c.dataset.name.includes(q)) {
                c.style.display = 'flex';
                visible++;
            } else {
                c.style.display = 'none';
            }
        });
        // Show empty state if needed
        let emptyDiv = document.querySelector('.tools-scroll .empty');
        if (!emptyDiv) {
            emptyDiv = document.createElement('div');
            emptyDiv.className = 'empty';
            emptyDiv.textContent = 'No matching tools';
            emptyDiv.style.display = 'none';
            document.getElementById('tools-container').appendChild(emptyDiv);
        }
        emptyDiv.style.display = visible === 0 ? 'block' : 'none';
    }

    // Make the function globally available for the oninput attribute
    window.filterTools = filterTools;

    // Add event listener to the search input
    const searchInput = document.getElementById('tool-search');
    if (searchInput) {
        searchInput.addEventListener('input', filterTools);
    }
})();
