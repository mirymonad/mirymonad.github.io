// Lazily load navigation list item element transitions

/** @type {HTMLElement[]} */
const anchorElements = document.querySelectorAll("a");

for (let anchorElement of anchorElements) {
    anchorElement.addEventListener("mouseover", () => {
        anchorElement.classList.add("rendered");
    });

    anchorElement.addEventListener("focus", () => {
        anchorElement.classList.add("rendered");
    });
}
