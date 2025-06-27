
document.addEventListener('DOMContentLoaded', () => {
    const abacusContainer = document.querySelector('.abacus');
    const number1Input = document.getElementById('number1');
    const number2Input = document.getElementById('number2');
    const addButton = document.getElementById('add-button');
    const subtractButton = document.getElementById('subtract-button');
    const multiplyButton = document.getElementById('multiply-button');
    const divideButton = document.getElementById('divide-button');
    const resultDisplay = document.getElementById('calculation-result');

    const NUM_RODS = 13;
    const UPPER_BEADS = 1;
    const LOWER_BEADS = 4;

    let abacusState = new Array(NUM_RODS).fill(0);

    function initializeAbacus() {
        abacusContainer.innerHTML = '';

        for (let i = 0; i < NUM_RODS; i++) {
            const rod = document.createElement('div');
            rod.classList.add('rod');
            rod.dataset.rodIndex = i;

            const upperDeck = document.createElement('div');
            upperDeck.classList.add('bead-set', 'bead-set-high');
            for (let j = 0; j < UPPER_BEADS; j++) {
                const bead = createBead(i, 'high', j, 5);
                upperDeck.appendChild(bead);
            }

            const lowerDeck = document.createElement('div');
            lowerDeck.classList.add('bead-set', 'bead-set-low');
            for (let j = 0; j < LOWER_BEADS; j++) {
                const bead = createBead(i, 'low', j, 1);
                lowerDeck.appendChild(bead);
            }

            rod.appendChild(upperDeck);
            rod.appendChild(lowerDeck);
            abacusContainer.appendChild(rod);
        }
        renderAbacus();
    }

    function createBead(rodIndex, set, beadIndex, value) {
        const bead = document.createElement('div');
        bead.classList.add('bead');
        bead.dataset.rodIndex = rodIndex;
        bead.dataset.beadSet = set;
        bead.dataset.beadIndex = beadIndex;
        bead.dataset.value = value;
        bead.addEventListener('click', handleBeadClick);
        return bead;
    }

    function renderAbacus() {
        for (let i = 0; i < NUM_RODS; i++) {
            const rodValue = abacusState[i];
            const upperBead = abacusContainer.querySelector(`.rod[data-rod-index='${i}'] .bead-set-high .bead`);
            const lowerBeads = abacusContainer.querySelectorAll(`.rod[data-rod-index='${i}'] .bead-set-low .bead`);

            if (rodValue >= 5) {
                upperBead.classList.add('active');
            } else {
                upperBead.classList.remove('active');
            }

            const lowerValue = rodValue % 5;
            lowerBeads.forEach((bead, index) => {
                if (index < lowerValue) {
                    bead.classList.add('active');
                } else {
                    bead.classList.remove('active');
                }
            });
        }
    }
    
    function setAbacusValue(value) {
        const strValue = Math.floor(Math.abs(value)).toString().padStart(NUM_RODS, '0');
        abacusState = strValue.split('').map(digit => parseInt(digit, 10));
        renderAbacus();
    }

    function handleBeadClick(event) {
        const bead = event.target;
        const rodIndex = parseInt(bead.dataset.rodIndex, 10);
        const beadSet = bead.dataset.beadSet;
        const beadValue = parseInt(bead.dataset.value, 10);
        let rodValue = abacusState[rodIndex];

        if (beadSet === 'high') {
             if (rodValue >= 5) {
                 rodValue -= 5;
             }
             else {
                rodValue += 5;
             }
        }
        else {
            const beadIndex = parseInt(bead.dataset.beadIndex, 10);
            const lowerValue = rodValue % 5;
            if (beadIndex < lowerValue ) {
                rodValue -= lowerValue - beadIndex;
            } else {
                rodValue += beadIndex - lowerValue + 1;
            }
        }
        
        abacusState[rodIndex] = rodValue;
        updateDisplayFromAbacus();
        renderAbacus();
    }
    
    function updateDisplayFromAbacus() {
        const currentValue = parseInt(abacusState.join(''), 10);
        resultDisplay.textContent = `Result: ${currentValue}`;
    }

    async function performCalculation(operation) {
        const num1 = parseFloat(number1Input.value);
        const num2 = parseFloat(number2Input.value);

        if (isNaN(num1) || isNaN(num2)) {
            displayError("Please enter valid numbers.");
            return;
        }

        try {
            const response = await fetch('/calculate', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ num1, num2, operation }),
            });

            const data = await response.json();

            if (data.success) {
                displayResult(data.result);
                setAbacusValue(data.result);
            } else {
                displayError(data.message || "An unknown error occurred.");
            }
        } catch (error) {
            console.error("Calculation error:", error);
            displayError("Failed to connect to the server.");
        }
    }

    function displayResult(result) {
        resultDisplay.textContent = `Result: ${result}`;
        resultDisplay.style.color = '#155724';
    }

    function displayError(message) {
        resultDisplay.textContent = `Error: ${message}`;
        resultDisplay.style.color = 'red';
    }

    function setupEventListeners() {
        addButton.addEventListener('click', () => performCalculation('add'));
        subtractButton.addEventListener('click', () => performCalculation('subtract'));
        multiplyButton.addEventListener('click', () => performCalculation('multiply'));
        divideButton.addEventListener('click', () => performCalculation('divide'));
        
        number1Input.addEventListener('input', () => setAbacusValue(number1Input.value));
    }

    function initializeApp() {
        initializeAbacus();
        setupEventListeners();
        setAbacusValue(0);
    }

    initializeApp();
});
