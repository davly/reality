package taxlot

import "errors"

// ErrNoLoss is returned by ApplyWashSale when the sale is not at a loss
// (proceeds >= cost basis). The wash-sale rule (IRC §1091) only disallows
// LOSSES; a gain sale is never a wash sale.
var ErrNoLoss = errors.New("taxlot: sale is not at a loss; wash-sale rule applies only to losses")

// ErrNonPositiveShares is returned when a share count that must be positive
// (the shares sold, or a replacement lot's share count) is zero or negative.
var ErrNonPositiveShares = errors.New("taxlot: share counts must be positive")

// ErrNegativeMoney is returned when a monetary input that must be
// non-negative (a cost basis or proceeds) is negative.
var ErrNegativeMoney = errors.New("taxlot: cost basis and proceeds must be non-negative")
